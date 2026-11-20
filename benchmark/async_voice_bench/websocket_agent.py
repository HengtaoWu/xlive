import base64
import asyncio
import json

import websockets


class WebSocketAgent:
    def __init__(self, socket):
        self.socket = socket

    async def send(self, kind, **fields):
        await self.socket.send(json.dumps(dict(type=kind, **fields)))

    async def start(self, context):
        await self.send('start', **context)
        ready = json.loads(await asyncio.wait_for(self.socket.recv(), timeout=60))
        if ready.get('type') != 'ready':
            raise ValueError('WebSocket agent must acknowledge session startup with {"type":"ready"}')

    async def send_audio(self, pcm):
        await self.send('input_audio', pcm=base64.b64encode(pcm).decode('ascii'))

    async def events(self):
        async for message in self.socket:
            event = json.loads(message)
            if event['type'] == 'audio':
                event['pcm'] = base64.b64decode(event['pcm'], validate=True)
            yield event

    async def tool_result(self, call_id, result):
        await self.send('tool_result', call_id=call_id, result=result)

    async def backend_event(self, event):
        await self.send('backend_event', event=event)

    async def playback(self, event):
        await self.send('playback', **event)

    async def aclose(self):
        await self.socket.close()


async def create_agent(config, runtime):
    socket = await websockets.connect(config['url'], max_size=16 * 1024 * 1024)
    return WebSocketAgent(socket)
