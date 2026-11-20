import asyncio
import json
import time
import wave

from .attribution import attribute_task
from .audio_agent import ModelFailure
from .jsonl_writer import JsonlWriter
from .protocol import write_json
from .realtime import ingest
from .sandbox import public_api_spec
from .tools_catalog import catalog_for_prompt


class TriggerPolicy:
    def __init__(self, episode):
        self.ep, self.dump = episode, {}

    def _goal_events(self, goal_id, event_type, t):
        final = self.dump['final_state']
        aliases = self.ep['environment'].get('materials', {}).get('aliases', {})
        return [event for event in self.dump['backend_events']
                if event['event'] == event_type and event['t'] <= t
                and attribute_task(final[event['task_id']]['tool'], final[event['task_id']]['versions'],
                                   self.ep['goals'], aliases) == goal_id]

    def _trigger_time_ok(self, turn, t, st):
        trg = turn.get("trigger", {"type": "after_agent_idle"})
        prev_end = st["prev_end"]
        if t < prev_end + 0.3:
            return False
        typ = trg["type"]
        ref_end = st["turn_end"].get(trg["ref"], prev_end) if trg.get("ref") else prev_end
        quiet = st["speaking"] is None and not st["queue"]
        if typ == "at":
            return t >= trg["t"]
        if typ == "after_turn":
            return t >= ref_end + trg.get("delay", 1.0)
        if typ == "after_agent_idle":
            gap, max_wait = trg.get("gap", 1.0), trg.get("max_wait", 12.0)
            q = quiet and (st["last_speech_end"] is None or t >= st["last_speech_end"] + gap)
            if (t >= ref_end + max_wait and q) or t >= ref_end + 3 * max_wait:
                return True
            spoke = st["last_speech_end"] is not None and st["last_speech_end"] > ref_end
            idle = quiet and t >= st["busy_until"]
            return spoke and idle and t >= st["last_speech_end"] + gap
        if typ == "on_event":
            evs = self._goal_events(trg["goal"], trg.get("event", "task_completed"), t)
            if evs:
                due = evs[0]["t"] + trg.get("delay", 0.5)
                if trg.get("barge_in"):
                    return t >= due
                return (t >= due and quiet) or t >= due + 15.0
            return t >= ref_end + trg.get("max_wait", 60.0)
        if typ == "after_agent_mention":
            evs = self._goal_events(trg["goal"], trg.get("event", "task_needs_input"), t)
            if evs:
                te = evs[0]["t"]
                gap = trg.get("gap", 1.0)
                spoke_after = st["last_speech_end"] is not None and st["last_speech_end"] > te
                if spoke_after and quiet and t >= st["last_speech_end"] + gap:
                    return True
                return quiet and t >= te + trg.get("max_wait", 30.0)
            return t >= ref_end + trg.get("fallback_s", 45.0)
        raise ValueError(f"unknown trigger type {typ}")


def dump_state(state):
    with state.lock:
        t = state.now()
        state.sb.advance(t)
        # Copy the records so the clock thread cannot mutate them during serialization.
        return json.loads(json.dumps(dict(t=t, turns=state.turns, sandbox_log=state.sb.log,
            backend_events=state.sb.events, final_state=state.sb.final_state(),
            episode_id=state.ep['episode_id'])))


def write_wav(path, pcm, rate):
    with wave.open(str(path), 'wb') as audio:
        audio.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
        audio.writeframes(pcm)


class AudioHarness:
    def __init__(self, agent, state, directory, output_rate):
        self.agent, self.state, self.directory = agent, state, directory
        self.output_rate = output_rate
        self.input_queue, self.output_queue = asyncio.Queue(), asyncio.Queue()
        self.tool_queue = asyncio.Queue()
        self.cancelled, self.ended, self.call_ids = set(), set(), set()
        self.speaking, self.last_end = False, None
        self.segments, self.current = [], None
        self.microphone = bytearray()
        self.assistant = bytearray()
        self.microphone_cursor = None
        self.playback_cursor = None
        self.event_log = JsonlWriter(directory / 'events.jsonl', mode='w')

    def record(self, event):
        self.event_log.record(dict(t=self.state.now(), **event))

    @staticmethod
    def place(target, pcm, start, rate):
        offset = round(start * rate) * 2
        if offset > len(target):
            target.extend(bytes(offset - len(target)))
        target[offset:offset + len(pcm)] = pcm

    async def receive(self):
        async for event in self.agent.events():
            kind = event['type']
            logged = {k: v for k, v in event.items() if k != 'pcm'}
            self.record(logged)
            if kind in ('audio', 'audio_end', 'cancel_audio'):
                identity = event['stream_id']
                if not isinstance(identity, str) or not identity:
                    raise ValueError('stream_id must be a nonempty string')
                if kind == 'cancel_audio':
                    self.cancelled.add(identity)
                    for segment in self.segments:
                        if segment['stream_id'] == identity and (self.speaking or identity not in self.ended):
                            segment['interrupted'] = True
                    await self.agent.playback(dict(event='cancelled', stream_id=identity, t=self.state.now()))
                    continue
                if identity in self.ended:
                    raise ValueError('Output after audio_end; use a new stream_id')
                if kind == 'audio':
                    pcm = event['pcm']
                    if not isinstance(pcm, bytes) or not pcm or len(pcm) % 2:
                        raise ValueError('audio.pcm must contain PCM16 bytes')
                    # Timestamp arrival before playback scheduling to avoid artificial gaps.
                    event = dict(event, _arrival_time=self.state.now())
                else:
                    self.ended.add(identity)
                self.output_queue.put_nowait(event)
            elif kind == 'tool_call':
                identity = event['call_id']
                if not isinstance(identity, str) or not identity or identity in self.call_ids:
                    raise ValueError('tool_call.call_id must be unique and nonempty')
                self.call_ids.add(identity)
                self.tool_queue.put_nowait(event)
            elif kind == 'error':
                failure_kind = event.get('failure_kind', 'infrastructure')
                if failure_kind not in ('model', 'infrastructure'):
                    raise ValueError('Unknown error.failure_kind')
                error_class = ModelFailure if failure_kind == 'model' else RuntimeError
                raise error_class(str(event.get('message', 'Agent error')))
            else:
                raise ValueError(f'Unknown adapter event: {kind}')
        raise RuntimeError('Agent connection closed before the episode ended')

    async def execute_tools(self):
        while True:
            event = await self.tool_queue.get()
            try:
                with self.state.lock:
                    result = self.state.sb.call(self.state.now(), event['name'], event['arguments'])
                await self.agent.tool_result(event['call_id'], result)
            finally:
                self.tool_queue.task_done()

    async def deliver_events(self):
        offset = 0
        while True:
            with self.state.lock:
                events = list(self.state.sb.events[offset:])
            for event in events:
                await self.agent.backend_event(event)
                self.record(dict(type='backend_event_delivered', event=event))
                offset += 1
            await asyncio.sleep(.02)

    async def microphone_loop(self):
        deadline = time.monotonic()
        while True:
            frame = self.input_queue.get_nowait() if not self.input_queue.empty() else None
            pcm = frame if frame is not None else bytes(640)
            start = self.state.now()
            await self.agent.send_audio(pcm)
            if self.microphone_cursor is None:
                self.microphone_cursor = round(start * 16000) * 2
            cursor = self.microphone_cursor
            if cursor > len(self.microphone):
                self.microphone.extend(bytes(cursor - len(self.microphone)))
            self.microphone[cursor:cursor + len(pcm)] = pcm
            self.microphone_cursor += len(pcm)
            deadline += .02
            if time.monotonic() - deadline > .1:
                raise RuntimeError('Agent microphone transport cannot sustain real-time input')
            await asyncio.sleep(max(0, deadline - time.monotonic()))
            if frame is not None:
                self.input_queue.task_done()

    async def player(self):
        frame_bytes = self.output_rate // 50 * 2
        while True:
            event = await self.output_queue.get()
            identity = event['stream_id']
            try:
                if event['type'] == 'audio_end':
                    await self.agent.playback(dict(event='finished', stream_id=identity,
                        interrupted=identity in self.cancelled, t=self.state.now()))
                    continue
                if identity in self.cancelled:
                    continue
                arrival = float(event['_arrival_time'])
                chunk_start = max(arrival, self.playback_cursor or arrival)
                if self.current is None or self.current['stream_id'] != identity:
                    self.current = dict(stream_id=identity, start=chunk_start,
                        end=chunk_start, pcm=bytearray(), interrupted=False)
                    self.segments.append(self.current)
                segment = self.current
                self.speaking = True
                await self.agent.playback(dict(event='started', stream_id=identity, t=self.state.now()))
                pcm = event['pcm']
                for offset in range(0, len(pcm), frame_bytes):
                    if identity in self.cancelled:
                        segment['interrupted'] = True
                        break
                    frame = pcm[offset:offset + frame_bytes]
                    start = chunk_start + offset / (2 * self.output_rate)
                    self.place(segment['pcm'], frame, start - segment['start'], self.output_rate)
                    self.place(self.assistant, frame, start, self.output_rate)
                    await asyncio.sleep(len(frame) / (2 * self.output_rate))
                    self.playback_cursor = start + len(frame) / (2 * self.output_rate)
                    segment['end'] = self.playback_cursor
                    self.last_end = self.state.now()
                await self.agent.playback(dict(event='chunk_played', stream_id=identity,
                    interrupted=identity in self.cancelled, t=self.state.now()))
            except asyncio.CancelledError:
                if self.speaking and self.current is not None:
                    self.current['interrupted'] = True
                raise
            finally:
                self.speaking = False
                self.output_queue.task_done()

    @staticmethod
    def read_user_audio(path):
        with wave.open(str(path), 'rb') as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
                raise ValueError('User audio must be mono 16 kHz PCM16')
            return audio.readframes(audio.getnframes())

    async def speak(self, turn, path):
        pcm = await asyncio.to_thread(self.read_user_audio, path)
        with self.state.lock:
            start = self.state.now()
            self.state.sb.set_anchor(turn['turn_id'], start)
            truth_start = self.state.sb.snapshot(start)
            self.state.starts[turn['turn_id']] = truth_start
        for offset in range(0, len(pcm), 640):
            self.input_queue.put_nowait(pcm[offset:offset + 640].ljust(640, b'\0'))
        await self.input_queue.join()
        with self.state.lock:
            end = self.state.now()
            self.state.turns.append(dict(turn_id=turn['turn_id'], start=start, end=end, text=turn['text'],
                backend_truth_start=truth_start, backend_truth=self.state.sb.snapshot(end)))
        return end

    async def save(self, transcriber, config):
        await self.event_log.aclose()
        write_wav(self.directory / 'microphone.wav', self.microphone, 16000)
        write_wav(self.directory / 'assistant.wav', self.assistant, self.output_rate)
        heard = []
        for index, segment in enumerate(self.segments):
            if not segment['pcm']:
                continue
            path = self.directory / f'heard-{index:03d}.wav'
            write_wav(path, segment['pcm'], self.output_rate)
            text = await transcriber(path, config)
            if not isinstance(text, str):
                raise TypeError('Transcriber must return the transcript string')
            heard.append(dict(start=segment['start'], end=segment['end'], text=text,
                interrupted=segment['interrupted'], audio_path=path.name))
        write_json(self.directory / 'heard_segments.json', heard)
        return heard


async def run_episode(episode, audio_paths, directory, state, agent, transcriber,
                      transcriber_config, system_id, max_seconds=420, output_rate=24000):
    await agent.start(dict(agent_context=episode['environment'].get('agent_context', {}),
        lang=episode['lang'], tools=public_api_spec(), catalog=catalog_for_prompt(),
        input_audio=dict(sample_rate=16000, channels=1, format='pcm_s16le', frame_ms=20),
        output_audio=dict(sample_rate=output_rate, channels=1, format='pcm_s16le')))
    with state.lock:
        state.reset()
    voice = AudioHarness(agent, state, directory, output_rate)
    policy = TriggerPolicy(episode)
    jobs = [asyncio.create_task(fn()) for fn in (voice.receive, voice.execute_tools,
        voice.deliver_events, voice.microphone_loop, voice.player)]
    triggers = dict(prev_end=0., turn_end={}, speaking=None, queue=[], last_speech_end=None, busy_until=0.)
    index, speech, final_end = 0, None, None
    failure = None
    reason = 'episode_deadline'
    try:
        while state.now() < max_seconds:
            for job in jobs:
                if job.done():
                    job.result()
                    raise RuntimeError('Audio worker stopped unexpectedly')
            policy.dump = dump_state(state)
            if speech is not None and speech.done():
                final_end = speech.result()
                triggers['prev_end'] = triggers['turn_end'][episode['turns'][index - 1]['turn_id']] = final_end
                speech = None
            triggers.update(speaking=True if voice.speaking else None,
                queue=[True] if not voice.output_queue.empty() else [], last_speech_end=voice.last_end)
            now = state.now()
            if speech is None and index < len(episode['turns']):
                turn = episode['turns'][index]
                if policy._trigger_time_ok(turn, now, triggers):
                    speech = asyncio.create_task(voice.speak(turn, audio_paths[turn['turn_id']]))
                    index += 1
            if index == len(episode['turns']) and speech is None and final_end is not None:
                last_event = max((event['t'] for event in policy.dump['backend_events']), default=final_end)
                quiet = not voice.speaking and voice.output_queue.empty() and (voice.last_end is None or now - voice.last_end >= 12)
                if now >= max(final_end + 90, last_event + 30) and quiet:
                    reason = 'post_user_observation_complete'
                    break
            await asyncio.sleep(.05)
    except Exception as exc:
        failure = dict(type=type(exc).__name__, message=str(exc),
            execution_status='model_failure' if isinstance(exc, ModelFailure) else 'infrastructure_failure')
    finally:
        for job in jobs + ([speech] if speech is not None else []):
            job.cancel()
        await asyncio.gather(*jobs, *([speech] if speech is not None else []), return_exceptions=True)
        await voice.event_log.aclose()
    dump = dump_state(state)
    write_json(directory / 'sandbox_dump.json', dump)
    heard = await voice.save(transcriber, transcriber_config)
    run = ingest(dump, heard, episode, system_id)
    run.update(execution_status=failure['execution_status'] if failure else 'complete', termination_reason=reason,
        harness_cfg=dict(mode='realtime', max_seconds=max_seconds, frame_ms=20, output_rate=output_rate,
            post_user_observation_seconds=90, post_event_grace_seconds=30,
            timestamp_granularity='played_utterance_segments'))
    if failure:
        run['failure'] = failure
    return run
