import argparse
import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from .sandbox import Sandbox, public_api_spec
from .dataset import load_episodes
from .tools_catalog import catalog_for_prompt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RTState:
    def __init__(self, ep):
        self.ep, self.lock = ep, threading.Lock()
        self._stop = threading.Event()
        self._clock = None
        self.reset()

    def reset(self):
        self.sb, self.t0, self.turns, self.starts = Sandbox(self.ep), time.monotonic(), [], {}

    def now(self):
        return time.monotonic() - self.t0

    def start_clock(self):
        def tick():
            while not self._stop.wait(.02):
                with self.lock:
                    self.sb.advance(self.now())
        self._clock = threading.Thread(target=tick, daemon=True)
        self._clock.start()

    def close(self):
        self._stop.set()
        if self._clock is not None:
            self._clock.join()


def make_handler(state):
    class H(BaseHTTPRequestHandler):
        def _send(self, obj, code=200):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path)
            with state.lock:
                t = state.now()
                state.sb.advance(t)
                if u.path == '/context':
                    return self._send({'agent_context': state.ep['environment'].get('agent_context', {}),
                                       'catalog': catalog_for_prompt(), 'tools': public_api_spec(), 'lang': state.ep['lang']})
                if u.path == "/events":
                    after = int(parse_qs(u.query).get("after", ["0"])[0])
                    evs = [e for e in state.sb.events if e["t"] <= t]
                    return self._send({"t": t, "events": evs[after:], "next": len(evs)})
                if u.path == "/dump":
                    return self._send({"t": t, "sandbox_log": state.sb.log, "backend_events": state.sb.events,
                                       "final_state": state.sb.final_state(), "turns": state.turns,
                                       'clock': {'origin_monotonic': state.t0, 'tick_seconds': .02},
                                       'episode_id': state.ep['episode_id']})
            self._send({"error": "not found"}, 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            if n < 0 or n > 2_000_000:
                return self._send({'error': 'invalid request size'}, 400)
            try:
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except (ValueError, UnicodeDecodeError):
                return self._send({'error': 'invalid JSON'}, 400)
            if not isinstance(body, dict):
                return self._send({'error': 'body must be an object'}, 400)
            with state.lock:
                p = urlparse(self.path).path
                if p == "/reset":
                    state.reset()
                    return self._send({"ok": True, 'origin_monotonic': state.t0, 't': 0.0})
                t = state.now()
                state.sb.advance(t)
                if p == "/tool":
                    return self._send(state.sb.call(t, body.get('name'), body.get("payload", {})))
                if p in ('/turn_start', '/turn_end'):
                    tid = body.get('turn_id')
                    if not isinstance(tid, str) or tid not in {u['turn_id'] for u in state.ep['turns']}:
                        return self._send({'error': 'unknown turn_id'}, 400)
                if p == "/turn_start":
                    if body['turn_id'] in state.starts:
                        return self._send({'error': 'duplicate turn_start'}, 400)
                    state.sb.set_anchor(body["turn_id"], t)
                    state.starts[body["turn_id"]] = state.sb.snapshot(t)
                    return self._send({"ok": True, "t": t})
                if p == "/turn_end":
                    if body['turn_id'] not in state.starts or any(u['turn_id'] == body['turn_id'] for u in state.turns):
                        return self._send({'error': 'missing start or duplicate turn_end'}, 400)
                    if any(type(body.get(k)) not in (int, float) or not math.isfinite(body[k]) for k in ('start', 'end')):
                        return self._send({'error': 'timestamps must be finite numbers'}, 400)
                    if not 0 <= body['start'] <= body['end']:
                        return self._send({'error': 'invalid turn interval'}, 400)
                    state.sb.advance(t)
                    state.turns.append({**body, "backend_truth": state.sb.snapshot(t),
                                        "backend_truth_start": state.starts.get(body["turn_id"], {})})
                    return self._send({"ok": True, "t": t})
            self._send({"error": "not found"}, 404)

        def log_message(self, *a):
            pass
    return H


def ingest(dump, agent_segments, episode, system_id):
    """dump: /dump JSON; agent_segments: [{start,end,text,interrupted?}] from timestamped ASR of system audio."""
    trace = []
    seen = set()
    known = {u['turn_id'] for u in episode['turns']}
    for u in dump['turns']:
        if u['turn_id'] in seen or u['turn_id'] not in known:
            raise ValueError('Duplicate or unknown user turn in recording')
        seen.add(u['turn_id'])
    for s in dump['turns'] + agent_segments:
        if any(type(s.get(k)) not in (int, float) or not math.isfinite(s[k]) for k in ('start', 'end')):
            raise ValueError('Audio timestamps must be finite numbers')
        if not 0 <= s['start'] <= s['end']:
            raise ValueError('Invalid audio interval')
    if dump.get('episode_id', episode['episode_id']) != episode['episode_id']:
        raise ValueError('Recording episode does not match selected episode')
    for u in dump["turns"]:
        trace.append({"type": "user_turn", "turn_id": u["turn_id"], "start": u["start"], "end": u["end"],
                      "text": u.get("text") or next(t["text"] for t in episode["turns"] if t["turn_id"] == u["turn_id"]),
                      "backend_truth": u["backend_truth"], "backend_truth_start": u.get("backend_truth_start", {})})
    for s in agent_segments:
        trace.append({"type": "agent_speech", "start": s["start"], "end": s["end"], "text": s["text"],
                      "heard_text": s["text"], "interrupted": bool(s.get("interrupted"))})
    for r in dump["sandbox_log"]:
        trace.append({"type": "tool_call", "t": r["t"], "name": r["op"], "payload": r["request"], "response": r["response"]})
    for e in dump["backend_events"]:
        trace.append({"type": "backend_event", **e})
    end_t = max([dump.get("t", 0)] + [x.get("end", x.get("t", 0)) for x in trace])
    played = {u["turn_id"] for u in dump["turns"]}
    return {"episode_id": episode["episode_id"], "system_id": system_id, "lang": episode["lang"],
            "timing_source": "external_realtime_recording",
            "harness_cfg": {"mode": "realtime"},
            "trace": sorted(trace, key=lambda e: e.get("t", e.get("start", 0))), "sandbox_log": dump["sandbox_log"],
            "backend_events": dump["backend_events"], "final_state": dump["final_state"],
            "unplayed_user_turns": [u["turn_id"] for u in episode["turns"] if u["turn_id"] not in played],
            "end_t": end_t}


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("serve"); s.add_argument("--episode", required=True); s.add_argument("--port", type=int, default=8765)
    i = sp.add_parser("ingest"); i.add_argument("--episode", required=True); i.add_argument("--dump", required=True)
    i.add_argument("--asr", required=True, help="jsonl of system speech segments {start,end,text,interrupted?}")
    i.add_argument("--system", required=True); i.add_argument("--out", required=True)
    a = ap.parse_args()
    ep = {e["episode_id"]: e for e in load_episodes(os.path.join(ROOT, "data", "episodes"))}[a.episode]
    if a.cmd == "serve":
        state = RTState(ep)
        state.start_clock()
        srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(state))
        print(f"sandbox for {ep['episode_id']} on :{a.port}")
        try:
            srv.serve_forever()
        finally:
            state.close()
            srv.server_close()
    else:
        dump = json.load(open(a.dump, encoding="utf-8"))
        segs = [json.loads(l) for l in open(a.asr, encoding="utf-8") if l.strip()]
        run = ingest(dump, segs, ep, a.system)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(run, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"run -> {a.out}")


if __name__ == "__main__":
    main()
