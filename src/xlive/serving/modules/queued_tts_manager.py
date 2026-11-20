from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from ...runtime.protocol_events import (
    ErrorOccurred, TTSChunkReady, TTSFinished, TTSStarted, TTSStopped,
    TTSTextSynthesized, LLMFirstSentence, TTSPaused, TTSResumed,
    VADSpeechStart, VADSpeechEnd,
)
from ...runtime.manager import Manager
from ...speech.events import XLiveUserCommitted, XLiveUserVocalizationDiscarded
from .tts_manager import TTSManager


@dataclass
class SpeechJob:
    generation_id: str
    utterance_id: str
    text: asyncio.Queue = field(default_factory=asyncio.Queue)
    audio: asyncio.Queue = field(default_factory=asyncio.Queue)
    played: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: bool = False
    producer: asyncio.Task | None = None
    consumer: asyncio.Task | None = None


class QueuedTTSManager(TTSManager):
    def __init__(self, event_bus, session_id, pipeline, config=None):
        super().__init__(event_bus, session_id, pipeline, config)
        self._jobs: dict[str, SpeechJob] = {}
        self._tail: SpeechJob | None = None
        self._synthesis_lock = asyncio.Lock()
        self._playback_lock = asyncio.Lock()
        self._start_permission = asyncio.Event()
        self._start_permission.set()
        self._speech_pause_task: asyncio.Task | None = None
        self._speech_paused = False

    @Manager.event_handler(VADSpeechStart, priority=140)
    async def _hold_for_user(self, event):
        if event.origin == "turn_detector":
            return
        self._start_permission.clear()
        if self._speech_pause_task:
            self._speech_pause_task.cancel()
        self._speech_pause_task = asyncio.create_task(self._pause_for_sustained_speech())

    async def _pause_for_sustained_speech(self):
        # Keep playing through brief sounds; buffer audio during sustained speech.
        await asyncio.sleep(0.65)
        if not self._start_permission.is_set() and self._playback_lock.locked():
            self._speech_paused = True
            await self._pause_tts()
            await self.event_bus.publish(TTSPaused(session_id=self.session_id), wait_for_completion=True)

    @Manager.event_handler(VADSpeechEnd, priority=140)
    async def _end_user_segment(self, event):
        if event.origin != "turn_detector" and self._speech_pause_task:
            self._speech_pause_task.cancel()
            self._speech_pause_task = None

    @Manager.event_handler(XLiveUserCommitted, priority=-10)
    @Manager.event_handler(XLiveUserVocalizationDiscarded, priority=-10)
    async def _release_user_hold(self, event):
        if self._speech_pause_task:
            self._speech_pause_task.cancel()
            self._speech_pause_task = None
        if self._speech_paused:
            self._speech_paused = False
            await self._resume_tts()
            await self.event_bus.publish(TTSResumed(session_id=self.session_id), wait_for_completion=True)
        self._start_permission.set()

    def _identity(self, job: SpeechJob) -> dict[str, str]:
        return dict(session_id=self.session_id, generation_id=job.generation_id,
                    utterance_id=job.utterance_id)

    def _job(self, event) -> SpeechJob | None:
        job = self._jobs.get(event.utterance_id)
        return job if job and job.generation_id == event.generation_id else None

    async def _handle_turn_tts_start(self, event):
        if not self._managed_identity_is_valid(event.generation_id, event.utterance_id):
            raise ValueError("TTS start requires a live utterance")
        if event.utterance_id in self._jobs:
            return
        job = SpeechJob(event.generation_id, event.utterance_id)
        previous = self._tail
        self._tail = job
        self._jobs[job.utterance_id] = job
        job.producer = asyncio.create_task(self._produce(job))
        job.consumer = asyncio.create_task(self._play(job, previous))

    async def _handle_turn_tts_append(self, event):
        job = self._job(event)
        if job and not job.cancelled and event.text:
            await job.text.put(event.text)

    async def _handle_turn_tts_flush(self, event):
        job = self._job(event)
        if job and not job.cancelled:
            await job.text.put(None)

    async def _produce(self, job: SpeechJob):
        try:
            # Queue text and audio per reply; synthesis proceeds independently of playback.
            async with self._synthesis_lock:
                buffer = ""
                first_sentence = True
                while not job.cancelled:
                    delta = await job.text.get()
                    final = delta is None
                    buffer += delta or ""
                    sentences, buffer = self._split_text_by_delimiters(buffer)
                    if final and buffer.strip():
                        sentences.append(buffer.strip())
                    if sentences and first_sentence:
                        first_sentence = False
                        await self.event_bus.publish(LLMFirstSentence(session_id=self.session_id))
                    for sentence in sentences:
                        model = self.pipeline.get_tts_model()
                        text = self._text_transform(sentence) if self._text_transform else sentence
                        duration = 0.0
                        async for pcm in model.async_synthesize_stream(text):
                            if pcm:
                                duration += self._chunk_duration_ms(pcm, model.sample_rate)
                                await job.audio.put((pcm, model.sample_rate))
                        if duration == 0:
                            raise RuntimeError("TTS synthesis returned no audio")
                        await self.event_bus.publish(TTSTextSynthesized(
                            session_id=self.session_id, text=sentence, audio_duration=duration,
                        ))
                    if final:
                        await job.audio.put(None)
                        return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail(job, exc)

    async def _play(self, job: SpeechJob, previous: SpeechJob | None):
        try:
            if previous:
                await previous.played.wait()
            if job.cancelled:
                return
            async with self._playback_lock:
                await self._start_permission.wait()
                if job.cancelled:
                    return
                self._generation_id, self._utterance_id = job.generation_id, job.utterance_id
                self._consumer_running = True
                await self._reset_outstanding_audio()
                await self.event_bus.publish(TTSStarted(**self._identity(job)), wait_for_completion=True)
                index = 0
                while not job.cancelled:
                    item = await job.audio.get()
                    if item is None:
                        self.output_controller.finish_synthesis(job.generation_id, job.utterance_id)
                        await self.event_bus.publish(TTSFinished(**self._identity(job)), wait_for_completion=True)
                        await job.played.wait()
                        return
                    pcm, sample_rate = item
                    if self.speed_controller and self.current_speed != 1.0:
                        pcm = await self.speed_controller.async_process(pcm, self.current_speed)
                    for chunk in self._split_audio_chunk(pcm, sample_rate):
                        if index == 0:
                            await self._start_permission.wait()
                        await self._resume_event.wait()
                        await self._wait_for_outstanding_budget()
                        if job.cancelled:
                            return
                        index += 1
                        chunk_id = f"{job.utterance_id}:{index}"
                        if not self.output_controller.admit_audio_chunk(job.generation_id, job.utterance_id, chunk_id):
                            return
                        await self._track_outstanding_chunk(self._chunk_duration_ms(chunk, sample_rate))
                        published = await self.event_bus.publish(TTSChunkReady(
                            **self._identity(job), chunk_id=chunk_id,
                            audio_chunk=chunk, sample_rate=sample_rate,
                        ), wait_for_completion=True)
                        if not published:
                            raise RuntimeError("failed to publish TTS audio")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail(job, exc)

    async def _handle_tts_playback_finished(self, event):
        job = self._job(event)
        if job and self.output_controller.confirm_playback_finished(event.generation_id, event.utterance_id):
            job.played.set()
            self._jobs.pop(job.utterance_id, None)

    async def _handle_turn_tts_stop(self, event):
        for job in list(self._jobs.values()):
            if job.generation_id == event.generation_id:
                await self._cancel_job(job)

    async def _cancel_job(self, job: SpeechJob):
        job.cancelled = True
        self.output_controller.cancel_generation(job.generation_id)
        tasks = [task for task in (job.producer, job.consumer)
                 if task is not None and task is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        # Send the stop event before releasing the next reply.
        await self.event_bus.publish(TTSStopped(**self._identity(job)), wait_for_completion=True)
        job.played.set()
        self._jobs.pop(job.utterance_id, None)

    async def _fail(self, job: SpeechJob, exc: Exception):
        await self._cancel_job(job)
        await self.event_bus.publish(ErrorOccurred(
            session_id=self.session_id, error_type="tts_generation_error", error_message=str(exc),
        ))

    async def _handle_turn_tts_pause(self, event):
        if self._event_matches_active_output(event):
            await self._pause_tts()
            await self.event_bus.publish(TTSPaused(session_id=self.session_id))

    async def _handle_turn_tts_resume(self, event):
        if self._event_matches_active_output(event):
            await self._resume_tts()
            await self.event_bus.publish(TTSResumed(session_id=self.session_id))

    async def reset_tts(self):
        if self._speech_pause_task:
            self._speech_pause_task.cancel()
            self._speech_pause_task = None
        for job in reversed(list(self._jobs.values())):
            await self._cancel_job(job)
        self._tail = None
        self._resume_event.set()
        self._speech_paused = False
        self._start_permission.set()

    async def shutdown(self):
        await self.reset_tts()
        await self.pipeline.get_tts_model().aclose()
