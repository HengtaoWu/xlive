import argparse
import asyncio
import json
import logging
import math
from http.server import ThreadingHTTPServer
from pathlib import Path
import threading

from .audio_agent import load_callable
from .audio_data import load_audio_index
from .audio_runner import run_episode
from .dataset import load_episodes
from .protocol import prepare_manifest, write_json
from .realtime import RTState, make_handler
from .score_programmatic import summarize


ROOT = Path(__file__).resolve().parents[1]


async def main():
    parser = argparse.ArgumentParser(description='Run duplex agents against AsyncVoiceBench')
    parser.add_argument('--agent', required=True, help='Async factory: module:callable(config, runtime) -> DuplexAgent')
    parser.add_argument('--agent-config', type=Path, required=True, help='Public JSON configuration; keep secrets in the environment')
    parser.add_argument('--transcriber', required=True, help='Async module:callable(wav_path, config) -> str')
    parser.add_argument('--transcriber-config', type=Path, required=True)
    parser.add_argument('--system-name', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--episode', nargs='+')
    parser.add_argument('--start-index', type=int, default=1)
    parser.add_argument('--max-seconds', type=float, default=420)
    parser.add_argument('--output-rate', type=int, default=24000)
    parser.add_argument('--programmatic-report', type=Path)
    parser.add_argument('--continue-on-error', action='store_true',
        help='Record isolated infrastructure failures and proceed to the next episode')
    parser.add_argument('--max-consecutive-errors', type=int, default=3,
        help='Stop after this many consecutive infrastructure failures')
    args = parser.parse_args()
    if args.max_consecutive_errors < 1:
        parser.error('Max consecutive errors must be positive')
    if args.start_index < 1 or (args.episode and args.start_index != 1):
        parser.error('Use a positive start index or explicit episode IDs')
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0 or args.output_rate < 8000 or args.output_rate % 50:
        parser.error('Invalid episode budget or output sample rate')
    factory, transcriber = load_callable(args.agent), load_callable(args.transcriber)
    logging.getLogger('websockets').setLevel(logging.WARNING)
    agent_config = json.loads(args.agent_config.read_text())
    transcriber_config = json.loads(args.transcriber_config.read_text())
    if not isinstance(agent_config, dict) or not isinstance(transcriber_config, dict):
        raise ValueError('Adapter configurations must be JSON objects')
    episodes = load_episodes(str(ROOT / 'data/episodes'))
    if args.start_index > len(episodes):
        parser.error('Start index exceeds episode count')
    episodes = episodes[args.start_index - 1:]
    if args.episode:
        selected = set(args.episode)
        episodes = [ep for ep in episodes if ep['episode_id'] in selected]
        if {ep['episode_id'] for ep in episodes} != selected:
            parser.error('Unknown episode ID')
    args.out = args.out.resolve()
    if args.programmatic_report:
        args.programmatic_report = args.programmatic_report.resolve()
        if args.programmatic_report == args.out or args.out in args.programmatic_report.parents:
            parser.error('Keep the aggregate report outside the run directory')
    rows = load_audio_index(ROOT)
    audio_paths = {}
    for row in rows:
        path = ROOT / row['out_path']
        if not path.is_file():
            raise FileNotFoundError(path)
        audio_paths.setdefault(row['episode_id'], {})[row['turn_id']] = path
    for ep in episodes:
        if set(audio_paths.get(ep['episode_id'], {})) != {turn['turn_id'] for turn in ep['turns']}:
            raise ValueError(f'Audio coverage mismatch: {ep["episode_id"]}')
    config = dict(agent=args.agent, agent_config=agent_config, transcriber=args.transcriber,
        transcriber_config=transcriber_config, max_seconds=args.max_seconds, output_rate=args.output_rate,
        continue_on_error=args.continue_on_error, max_consecutive_errors=args.max_consecutive_errors)
    manifest = prepare_manifest(args.out, episodes, args.system_name, config, 'external_realtime_recording')
    consecutive_errors = 0
    for episode in episodes:
        eid = episode['episode_id']
        path = args.out / (eid + '.json')
        if path.exists():
            print(f'SKIP recorded attempt: {eid}', flush=True)
            continue
        directory = args.out / 'cases' / eid
        directory.mkdir(parents=True, exist_ok=False)
        state, agent = RTState(episode), None
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(state))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        state.start_clock()
        print(f'START {eid}', flush=True)
        try:
            agent = await factory(agent_config, dict(case_dir=str(directory),
                sandbox_url=f'http://127.0.0.1:{server.server_port}'))
            run = await run_episode(episode, audio_paths[eid], directory, state, agent,
                transcriber, transcriber_config, args.system_name, args.max_seconds, args.output_rate)
        except Exception as exc:
            run = dict(episode_id=eid, system_id=args.system_name,
                timing_source='external_realtime_recording',
                execution_status='infrastructure_failure', failure=dict(type=type(exc).__name__, message=str(exc)))
            write_json(directory / 'exception.json', run['failure'])
        finally:
            if agent is not None:
                try:
                    await agent.aclose()
                except Exception as exc:
                    run.update(execution_status='infrastructure_failure',
                        failure=dict(type=type(exc).__name__, message=str(exc), phase='adapter_cleanup'))
            await asyncio.to_thread(server.shutdown)
            state.close()
            server.server_close()
            thread.join()
        write_json(path, run)
        print(f'FINISH {eid}: {run["execution_status"]}', flush=True)
        if args.programmatic_report:
            summarize(args.out, args.programmatic_report)
        if run['execution_status'] == 'infrastructure_failure':
            consecutive_errors += 1
            if not args.continue_on_error or consecutive_errors >= args.max_consecutive_errors:
                raise RuntimeError(f'Infrastructure failure recorded; stopping after {consecutive_errors} consecutive errors')
            print(f'CONTINUE after recorded failure: {eid}: {run.get("failure", {})}', flush=True)
        else:
            consecutive_errors = 0


if __name__ == '__main__':
    asyncio.run(main())
