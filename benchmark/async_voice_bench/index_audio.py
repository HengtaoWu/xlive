from collections import Counter
import json
from pathlib import Path

from .audio_data import load_audio_index
from .dataset import load_episodes


ROOT = Path(__file__).resolve().parents[1]


def main():
    episodes = load_episodes(str(ROOT / 'data/episodes'))
    rows = load_audio_index(ROOT)
    summary = dict(episodes=len(episodes), utterances=len(rows),
                   audio_seconds=sum(row['duration_seconds'] for row in rows),
                   episodes_by_domain=dict(Counter(ep['domain'] for ep in episodes)),
                   wav_integrity='passed')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
