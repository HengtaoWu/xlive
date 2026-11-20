import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from queue import Full, Queue


class JsonlWriter:
    def __init__(self, path, mode='a'):
        self.path, self.mode = Path(path), mode
        self.queue = Queue(maxsize=4096)
        self.closed = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='benchmark-jsonl')
        self.worker = self.executor.submit(self._write)

    def _write(self):
        with self.path.open(self.mode, encoding='utf-8') as stream:
            while True:
                line = self.queue.get()
                if line is None:
                    return
                stream.write(line)
                stream.flush()

    def record(self, value):
        if self.closed:
            raise RuntimeError('Cannot append to a closed JSONL writer')
        if self.worker.done():
            self.worker.result()
            raise RuntimeError('JSONL writer stopped unexpectedly')
        # Serialize when the event occurs so timestamps and data snapshots are unaffected by disk latency.
        self.queue.put_nowait(json.dumps(value, ensure_ascii=False) + '\n')

    def close(self):
        if self.closed:
            self.worker.result()
            return
        self.closed = True
        try:
            while not self.worker.done():
                try:
                    self.queue.put(None, timeout=.1)
                    break
                except Full:
                    continue
            self.worker.result()
        finally:
            self.executor.shutdown(wait=True)

    async def aclose(self):
        await asyncio.to_thread(self.close)
