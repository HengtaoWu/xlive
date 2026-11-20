# X-Live Web

The browser client handles microphone input, streamed speech playback,
conversation history, and background tasks. It includes an audio-reactive orb,
a Chinese/English voice selector, and draggable task-history panels.

## Development

```bash
cd src/web
npm ci
npm run dev
```

Open `http://127.0.0.1:8000`. Live conversations require the X-Live runtime on
port 8001.

## Check and build

```bash
npm run check:app
npm run build
```

For the full system, run `setup.sh` and `start.sh` from the repository root.
