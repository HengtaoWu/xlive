"""Versioned asynchronous task sandbox with episode-controlled timing."""
import copy

from .tools_catalog import TOOLS, EXECUTORS, PARTIAL_CAPABLE, validate_args, validate_materials, canon


class Task:
    def __init__(self, tid, tool, args, t, beh):
        self.id, self.tool, self.beh = tid, tool, beh
        self.version = 1
        self.versions = [{"v": 1, "t": round(t, 3), "args": copy.deepcopy(args)}]
        self.status = "running"
        self.run_start = t
        self.due = t
        self.anchor = beh.get("release_not_before")
        self.partial, self.partial_sent = None, False
        self.result, self.result_version, self.needs, self.error = None, None, None, None

    @property
    def args(self):
        return self.versions[-1]["args"]


class Sandbox:
    def __init__(self, episode):
        env = episode["environment"]
        self.materials = env.get("materials", {})
        self.behaviors = env.get("tool_behaviors", {})
        for tool in self.behaviors:
            if tool not in TOOLS:
                raise ValueError(f"Unknown configured tool: {tool}")
            error = validate_materials(tool, self.materials)
            if error:
                raise ValueError(f"Invalid episode environment: {error}")
        self.aliases = self.materials.get("aliases", {})
        self.tasks, self.events, self.log = {}, [], []
        self.anchors, self._n, self._execs = {}, 0, {}

    def set_anchor(self, turn_id, t):
        self.anchors.setdefault(turn_id, t)

    def _duration(self, tool, args, amended=False):
        beh = self.behaviors.get(tool, {})
        d = beh.get("duration_s", 10.0)
        for ov in beh.get("overrides", []):
            if all(canon(args.get(k), self.aliases) == canon(v, self.aliases) for k, v in ov["when"].items()):
                d = ov["set"].get("duration_s", d)
        if amended:
            d = beh.get("amend_policy", {}).get("duration_s", d)
        return d

    def _release_ok(self, task, t):
        anc = task.anchor
        if not anc:
            return True
        start = self.anchors.get(anc["turn"])
        if start is not None:
            return t >= start + anc.get("offset", 0.0)
        return t >= task.due + anc.get("max_hold_s", 30.0)

    def call(self, t, name, payload):
        self.advance(t)
        if payload is None:
            payload = {}
        fn = {"submit_task": self._submit, "amend_task": self._amend, "get_task": self._get,
              "cancel_task": self._cancel, "list_tasks": self._list}.get(name) if isinstance(name, str) else None
        if not isinstance(payload, dict):
            resp = {"ok": False, "error": "payload must be an object"}
        elif name in ("amend_task", "get_task", "cancel_task") and not isinstance(payload.get("task_id"), str):
            resp = {"ok": False, "error": "task_id must be a string"}
        else:
            resp = fn(t, payload) if fn else {"ok": False, "error": f"unknown operation '{name}'"}
        self.log.append({"t": round(t, 3), "op": name, "request": copy.deepcopy(payload), "response": copy.deepcopy(resp)})
        return resp

    def _submit(self, t, p):
        tool, args = p.get("tool"), p.get("args", {})
        err = validate_args(tool, args)
        if err:
            return {"ok": False, "error": err}
        err = validate_materials(tool, self.materials)
        if err:
            return {"ok": False, "error": err, "error_code": "tool_unavailable"}
        self._n += 1
        tid = f"T{self._n}"
        task = Task(tid, tool, args, t, self.behaviors.get(tool, {}))
        task.due = t + self._duration(tool, args)
        self.tasks[tid] = task
        return {"ok": True, "task_id": tid, "status": "running", "version": 1}

    def _task(self, p):
        return self.tasks.get(p.get("task_id"))

    def _amend(self, t, p):
        task = self._task(p)
        if task is None:
            return {"ok": False, "error": "unknown task_id"}
        if task.status == "cancelled":
            return {"ok": False, "error": "task was cancelled"}
        changes = p.get("changes")
        if not isinstance(changes, dict) or not changes:
            return {"ok": False, "error": "changes must be a non-empty object"}
        new = copy.deepcopy(task.args)
        for k, v in changes.items():
            if v is None:
                new.pop(k, None)
            else:
                new[k] = v
        err = validate_args(task.tool, new)
        if err:
            return {"ok": False, "error": err}
        task.version += 1
        task.versions.append({"v": task.version, "t": round(t, 3), "args": new})
        task.status, task.run_start = "running", t
        task.due = t + self._duration(task.tool, new, amended=True)
        task.anchor = task.beh.get("amend_policy", {}).get("release_not_before")
        task.needs = task.error = None
        task.partial, task.partial_sent = None, False
        task.result, task.result_version = None, None
        return {"ok": True, "task_id": task.id, "status": "running", "version": task.version}

    def _view(self, task, t):
        v = {"task_id": task.id, "tool": task.tool, "status": task.status, "version": task.version,
             "args": copy.deepcopy(task.args)}
        if task.status == "running":
            span = max(1e-6, task.due - task.run_start)
            frac = max(0.0, min(0.99, (t - task.run_start) / span))
            v["progress_pct"] = int(frac * 100)
            stages = [s for s in task.beh.get("stages", []) if s["at_frac"] <= frac]
            if stages:
                v["stage"] = stages[-1]["label"]
            if task.partial_sent:
                v["partial_result"] = copy.deepcopy(task.partial)
        elif task.status == "completed":
            v["result"] = copy.deepcopy(task.result)
        elif task.status == "needs_input":
            v["needs_input"] = task.needs
        elif task.status == "failed":
            v["error"] = task.error
        return v

    def _get(self, t, p):
        task = self._task(p)
        if task is None:
            return {"ok": False, "error": "unknown task_id"}
        return {"ok": True, **self._view(task, t)}

    def _cancel(self, t, p):
        task = self._task(p)
        if task is None:
            return {"ok": False, "error": "unknown task_id"}
        if task.status in ("completed", "cancelled", "failed"):
            return {"ok": False, "error": f"task already {task.status}"}
        task.status = "cancelled"
        return {"ok": True, "task_id": task.id, "status": "cancelled"}

    def _list(self, t, p):
        return {"ok": True, "tasks": [self._view(x, t) for x in self.tasks.values()]}

    def advance(self, t):
        for task in self.tasks.values():
            if task.status != "running":
                continue
            p = task.beh.get("partial")
            if (p and task.tool in PARTIAL_CAPABLE and not task.partial_sent and task.version == 1
                    and t >= task.run_start + p["at_s"]):
                task.partial_sent = True
                task.partial = EXECUTORS[task.tool](task.args, self.materials, t_rel=p["at_s"], final=False)
                self._emit(task.run_start + p["at_s"], task, "task_partial", {"partial_result": task.partial})
            if t >= task.due and self._release_ok(task, t):
                n = self._execs.get(task.tool, 0)
                self._execs[task.tool] = n + 1
                if n < task.beh.get("fail_first_n", 0):
                    task.status, task.error = "failed", task.beh.get("fail_error", "backend error")
                    self._emit(t, task, "task_failed", {"error": task.error})
                    continue
                res = EXECUTORS[task.tool](task.args, self.materials)
                if "__needs_input__" in res:
                    task.status, task.needs = "needs_input", res["__needs_input__"]
                    self._emit(t, task, "task_needs_input", {"needs_input": task.needs})
                elif "__error__" in res:
                    task.status, task.error = "failed", res["__error__"]
                    self._emit(t, task, "task_failed", {"error": task.error})
                else:
                    task.status, task.result, task.result_version = "completed", res, task.version
                    self._emit(t, task, "task_completed", {"result": res})

    def _emit(self, t, task, etype, payload):
        ev = {"t": round(t, 3), "event": etype, "task_id": task.id, "tool": task.tool,
              "version": task.version, **copy.deepcopy(payload)}
        self.events.append(ev)

    def snapshot(self, t):
        return {k: self._view(v, t) for k, v in self.tasks.items()}

    def final_state(self):
        return {k: {"tool": v.tool, "status": v.status, "versions": v.versions, "result": v.result,
                    "result_version": v.result_version, "needs": v.needs, "error": v.error}
                for k, v in self.tasks.items()}


def public_api_spec():
    tools_enum = list(TOOLS)
    return [
        {"type": "function", "function": {"name": "submit_task", "description":
            "Start a background task. Returns a task_id immediately; the result arrives later as a backend event.",
            "parameters": {"type": "object", "properties": {"tool": {"type": "string", "enum": tools_enum},
                                                            "args": {"type": "object"}}, "required": ["tool", "args"]}}},
        {"type": "function", "function": {"name": "amend_task", "description":
            "Change requirements of an existing task. 'changes' is merged into its args (null removes a key); the task reruns.",
            "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}, "changes": {"type": "object"}},
                           "required": ["task_id", "changes"]}}},
        {"type": "function", "function": {"name": "get_task", "description": "Get the live status/progress/result of a task.",
            "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}}},
        {"type": "function", "function": {"name": "cancel_task", "description": "Cancel a running task.",
            "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}}},
        {"type": "function", "function": {"name": "list_tasks", "description": "List all tasks and their status.",
            "parameters": {"type": "object", "properties": {}}}},
    ]
