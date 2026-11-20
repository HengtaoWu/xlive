from pathlib import Path
import wave

from .dataset import load_episodes


def load_audio_index(root):
    root = Path(root).resolve()
    audio_root = root / 'data/audio'
    rows, seen, paths = [], set(), set()
    episodes = load_episodes(str(root / 'data/episodes'))
    if not episodes:
        raise ValueError('No benchmark episodes found')
    for ep in episodes:
        for turn in ep['turns']:
            key = (ep['episode_id'], turn['turn_id'])
            if key in seen:
                raise ValueError(f'Duplicate utterance: {key}')
            relative = Path('data/audio') / key[0] / (key[1] + '.wav')
            path = (root / relative).resolve()
            if not path.is_relative_to(audio_root) or path in paths:
                raise ValueError(f'Invalid or duplicate recording path: {key}')
            with wave.open(str(path), 'rb') as audio:
                if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
                    raise ValueError(f'Expected mono 16 kHz PCM16: {path}')
                frames = audio.getnframes()
                pcm = audio.readframes(frames)
                if frames <= 0 or len(pcm) != frames * 2:
                    raise ValueError(f'Empty or truncated WAV: {path}')
            rows.append(dict(episode_id=key[0], turn_id=key[1], utt_id=f'{key[0]}__{key[1]}',
                text=turn['text'], split=ep['split'], lang=ep['lang'], domain=ep['domain'],
                template_id=ep['template_id'], out_path=relative.as_posix(), sample_rate=16000,
                duration_seconds=frames / 16000))
            seen.add(key)
            paths.add(path)
    inventory = list(audio_root.rglob('*.wav'))
    if len(inventory) != len(paths) or {path.resolve() for path in inventory} != paths:
        raise ValueError('WAV inventory differs from episode turns')
    return rows
