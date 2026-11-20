"""Score observable task effects; amendments and cancel/resubmit can satisfy modifications."""
from .attribution import attribute_task, check_args, value_ok, _MISSING
from .textutil import match_groups

COLLAPSE_WINDOW_S = 20.0


def _turn_times(trace):
    return {e["turn_id"]: (e["start"], e["end"]) for e in trace if e["type"] == "user_turn"}


def heard_segments(trace):
    segs = [e for e in trace if e["type"] == "agent_speech" and e["heard_text"].strip()]
    return sorted(segs, key=lambda s: s["start"])


def observed_ops(ep, run):
    aliases = ep["environment"].get("materials", {}).get("aliases", {})
    final = run["final_state"]
    goal_of = {tid: attribute_task(v["tool"], v["versions"], ep["goals"], aliases) for tid, v in final.items()}
    raw = []
    for rec in run["sandbox_log"]:
        resp, req = rec["response"], rec["request"]
        if not resp.get("ok"):
            continue
        if rec["op"] == "submit_task":
            tid = resp["task_id"]
            raw.append({"op": "start", "t": rec["t"], "task": tid, "goal": goal_of.get(tid), "args": req["args"]})
        elif rec["op"] == "amend_task":
            tid = req["task_id"]
            version = resp.get("version")
            state = next((v["args"] for v in final[tid]["versions"] if v["v"] == version), {})
            before = next((v["args"] for v in final[tid]["versions"] if v["v"] == version - 1), {})
            raw.append({"op": "modify", "t": rec["t"], "task": tid, "goal": goal_of.get(tid),
                        "fields": sorted((req.get("changes") or {}).keys()), "args": state,
                        "previous_args": before, "via": "amend"})
        elif rec["op"] == "cancel_task":
            tid = req["task_id"]
            raw.append({"op": "cancel", "t": rec["t"], "task": tid, "goal": goal_of.get(tid)})
    # collapse cancel+resubmit (either order) of the same goal into one modify
    ops, used = [], set()
    for i, a in enumerate(raw):
        if i in used:
            continue
        if a["op"] in ("cancel", "start") and a["goal"]:
            for j in range(i + 1, len(raw)):
                b = raw[j]
                if j in used or b["goal"] != a["goal"] or b["t"] - a["t"] > COLLAPSE_WINDOW_S:
                    continue
                if {a["op"], b["op"]} == {"cancel", "start"} and a["task"] != b["task"]:
                    st = a if a["op"] == "start" else b
                    ca = b if st is a else a
                    prior_start = next((r for r in raw if r["op"] == "start" and r["task"] == ca["task"]), None)
                    if prior_start is None or prior_start is st:
                        continue
                    old = [v["args"] for v in final[ca["task"]]["versions"] if v["t"] <= ca["t"]][-1]
                    fields = sorted(k for k in set(old) | set(st["args"]) if old.get(k) != st["args"].get(k))
                    ops.append({"op": "modify", "t": b["t"], "task": st["task"], "goal": a["goal"],
                                "fields": fields, "args": st["args"], "previous_args": old, "via": "cancel+resubmit"})
                    used.update({i, j})
                    break
        if i in used:
            continue
        ops.append(a)
        used.add(i)
    return sorted(ops, key=lambda o: o["t"]), goal_of


def _op_matches(expected_op, observed_op):
    if expected_op == "retry":
        return observed_op in ("start", "modify")
    return expected_op == observed_op


def tool_selection(ep, run, *, enforce_deadlines=False):
    tt = _turn_times(run["trace"])
    ops, goal_of = observed_ops(ep, run)
    expected = []
    for g in ep["goals"]:
        for k, e in enumerate(g.get("ops", [])):
            expected.append({**e, "goal": g["goal_id"], "key": f"{g['goal_id']}#{k}"})
    matched_obs, details, absorbed_idx = set(), [], set()
    first_start = {}
    for i, o in enumerate(ops):
        if o["op"] == "start" and o["goal"] and o["goal"] not in first_start:
            first_start[o["goal"]] = (i, o)
    for e in expected:
        lo = tt.get(e["after"], (None, None))[0]
        hi = (tt.get(e["after"], (0, 0))[1] or 0) + e.get("deadline_s", 30.0)
        upper = hi if enforce_deadlines else float('inf')
        hit = None
        if lo is not None:
            for i, o in enumerate(ops):
                if i in matched_obs or o["goal"] != e["goal"] or not _op_matches(e["op"], o["op"]):
                    continue
                if not (lo - 0.01 <= o["t"] <= upper):
                    continue
                if e["op"] == "modify" and o["op"] == "modify" and not set(e.get("fields", [])) <= set(o.get("fields", [])):
                    continue
                hit = i
                break
            if hit is None and e["op"] == "modify" and e["goal"] in first_start:
                i, o = first_start[e["goal"]]
                if lo <= o["t"] <= upper and all(f in o["args"] for f in e.get("fields", [])):
                    hit = ("absorbed", i)
        if isinstance(hit, tuple):
            absorbed_idx.add(hit[1])
            details.append({"expected": e["key"], "op": e["op"], "matched": True, "how": "absorbed_in_start"})
        elif hit is not None:
            matched_obs.add(hit)
            details.append({"expected": e["key"], "op": e["op"], "matched": True, "t": ops[hit]["t"],
                            "latency_s": round(ops[hit]["t"] - tt[e["after"]][1], 2), "via": ops[hit].get("via")})
        else:
            details.append({"expected": e["key"], "op": e["op"], "matched": False})
        if hit is not None:
            idx = hit[1] if isinstance(hit, tuple) else hit
            details[-1].update(t=ops[idx]['t'],
                latency_s=round(ops[idx]['t'] - tt[e['after']][1], 2),
                deadline_s=e.get('deadline_s', 30.0),
                on_time=ops[idx]['t'] <= hi,
                lateness_s=round(max(0.0, ops[idx]['t'] - hi), 3))
        if hit is not None and "constraints" in e:
            idx = hit[1] if isinstance(hit, tuple) else hit
            args = ops[idx].get("args", {})
            constraints, allowed = e["constraints"], e.get("allowed_extra", [])
            # Late starts use the latest instructions completed before dispatch.
            if ops[idx]["op"] == "start":
                later = [x for x in expected if x["goal"] == e["goal"] and "constraints" in x and x["after"] in tt
                         and tt[x["after"]][1] <= ops[idx]["t"]
                         and tt[x["after"]][1] > tt[e["after"]][1]]
                if later:
                    latest = max(later, key=lambda x: tt[x["after"]][1])
                    constraints, allowed = latest["constraints"], latest.get("allowed_extra", [])
            aliases = ep["environment"].get("materials", {}).get("aliases", {})
            n, total, errors = check_args(args, constraints, aliases, allowed)
            if ops[idx]["op"] == "modify":
                previous = ops[idx].get("previous_args", {})
                for field in set(previous) | set(args):
                    if field in e.get("allowed_changes", e.get("fields", [])):
                        continue
                    before, after = previous.get(field, _MISSING), args.get(field, _MISSING)
                    unchanged = (before is after if before is _MISSING or after is _MISSING
                                 else value_ok(after, before, aliases))
                    if not unchanged:
                        total += 1
                        errors.append({"slot": field, "expected": "preserve unmodified argument",
                                       "actual": args.get(field)})
            details[-1].update(args_exact=not errors, arg_errors=errors,
                               arg_score=n / total if total else 1.0)
    tp = sum(1 for d in details if d["matched"])
    fn = len(details) - tp
    fps = [o for i, o in enumerate(ops) if i not in matched_obs and i not in absorbed_idx]
    fp = len(fps)
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn, "expected": details,
            "false_positive_ops": [{k: o[k] for k in ("op", "t", "goal", "task")} for o in fps],
            "observed_ops": ops, "task_goal": goal_of}


def arg_accuracy(ep, run, goal_of):
    aliases = ep["environment"].get("materials", {}).get("aliases", {})
    final = run["final_state"]
    out = {}
    for g in ep["goals"]:
        fin = g["final"]
        tids = [t for t, gid in goal_of.items() if gid == g["goal_id"]]
        if fin["status"] == "cancelled":
            ok = bool(tids) and all(final[t]["status"] == "cancelled" for t in tids)
            out[g["goal_id"]] = {"scored": False, "status_ok": ok, "statuses": {t: final[t]["status"] for t in tids}}
            continue
        live = [t for t in tids if final[t]["status"] not in ("cancelled", "failed")] or \
               [t for t in tids if final[t]["status"] != "cancelled"]
        if not live:
            out[g["goal_id"]] = {"scored": True, "score": 0.0, "exact": False, "errors": ["no live task for goal"],
                                 "status_ok": False}
            continue
        t = live[-1]
        args = final[t]["versions"][-1]["args"]
        ok, total, errs = check_args(args, fin.get("constraints", {}), aliases, fin.get("allowed_extra", []))
        st_ok = final[t]["status"] == fin.get("status", "completed")
        out[g["goal_id"]] = {"scored": True, "score": round(ok / total, 4) if total else 1.0, "exact": not errs,
                             "errors": errs, "task": t, "status": final[t]["status"], "status_ok": st_ok,
                             "duplicates_live": len(live) - 1}
    scored = [v["score"] for v in out.values() if v["scored"]]
    return {"mean": round(sum(scored) / len(scored), 4) if scored else None, "goals": out}


def deliveries(ep, run, goal_of):
    segs = heard_segments(run["trace"])
    tt = _turn_times(run["trace"])
    events = run["backend_events"]
    res = {}
    for g in ep["goals"]:
        tids = {t for t, gid in goal_of.items() if gid == g["goal_id"]}
        d = g.get("delivery")
        entry = {}
        if d and d.get("required"):
            comp = [e for e in events if e["task_id"] in tids and e["event"] == "task_completed"]
            if not comp:
                entry["delivered"] = False
                entry["reason"] = "no completed result"
            else:
                tc = comp[-1]["t"]
                hits = [s for s in segs if s["start"] >= tc - 0.01
                        and match_groups(s["heard_text"], d["key_facts"], d.get("min_facts", 1))]
                entry["delivered"] = bool(hits)
                entry["result_t"] = tc
                if hits:
                    entry["delay_s"] = round(hits[0]["start"] - tc, 2)
                    entry["first_segment_start"] = hits[0]["start"]
                asks = [tt[u][1] for u in g.get("asked_in_turns", []) if u in tt]
                rep = 0
                for a, b in zip(hits, hits[1:]):
                    if not any(a["start"] < x <= b["start"] for x in asks):
                        rep += 1
                entry["unprompted_repeats"] = rep
        p = g.get("problem_report")
        if p:
            pev = [e for e in events if e["task_id"] in tids and e["event"] == p["event"]]
            if not pev:
                entry["problem_reported"] = None
            else:
                tp = pev[0]["t"]
                hits = [s for s in segs if s["start"] >= tp - 0.01 and match_groups(s["heard_text"], p["key_facts"])]
                entry["problem_reported"] = bool(hits)
                entry["problem_mentions"] = len(hits)
                if hits:
                    entry["problem_delay_s"] = round(hits[0]["start"] - tp, 2)
        res[g["goal_id"]] = entry
    return res


def _busy(snapshot):
    return any(v.get("status") in ("running", "needs_input") for v in snapshot.values())


def timing_layer(ep, run, ops=None):
    """Measure per-turn response gaps, dispatch, pending work, and speech overlap."""
    trace = run["trace"]
    turns = sorted([e for e in trace if e["type"] == "user_turn"], key=lambda e: e["start"])
    segs = sorted([e for e in trace if e["type"] == "agent_speech"], key=lambda e: e["start"])
    if ops is None:
        ops, _ = observed_ops(ep, run)
    out = {}
    for i, u in enumerate(turns):
        nxt_start = turns[i + 1]["start"] if i + 1 < len(turns) else float("inf")
        resp = [s for s in segs if u["end"] - 0.01 <= s["start"] < nxt_start]
        gap = round(resp[0]["start"] - u["end"], 2) if resp else None
        dispatch = any(o["op"] in ("start", "modify") and u["end"] - 0.01 <= o["t"] < min(nxt_start, u["end"] + 20)
                       for o in ops)
        out[u["turn_id"]] = {
            "response_gap_s": gap,
            "dispatch": dispatch,
            "busy": _busy(u.get("backend_truth", {})) or _busy(u.get("backend_truth_start", {})),
            "talk_over": sum(1 for s in segs if u["start"] + 0.2 < s["start"] < u["end"]),
            "overlap_seconds": round(sum(max(0.0, min(s["end"], u["end"]) - max(s["start"], u["start"]))
                                          for s in segs), 3),
            "barge_in_stop_delays_s": [round(s["end"] - u["start"], 3)
                                        for s in segs if s["start"] < u["start"] < s["end"]],
            "silence_until_response_s": gap if gap is not None else round(min(nxt_start, run.get("end_t", u["end"])) - u["end"], 2),
        }
    first_audio = [v["response_gap_s"] for v in out.values() if v["response_gap_s"] is not None]
    dispatch_gaps = [v["silence_until_response_s"] for v in out.values() if v["dispatch"]]
    return {
        "turns": out,
        "talk_over": sum(v["talk_over"] for v in out.values()),
        "overlap_seconds": round(sum(v["overlap_seconds"] for v in out.values()), 3),
        "interrupted_segments": sum(1 for s in segs if s.get("interrupted")),
        "dropped_planned_speech": sum(1 for e in trace if e["type"] == "agent_speech_dropped"),
        "first_audio_after_user_s": first_audio,
        "dispatch_silence_s": dispatch_gaps,
        "status_queries": sum(1 for r in run["sandbox_log"] if r["op"] in ("get_task", "list_tasks")),
        "rejected_calls": sum(1 for r in run["sandbox_log"] if not r["response"].get("ok")),
    }


def programmatic(ep, run):
    ts = tool_selection(ep, run)
    aa = arg_accuracy(ep, run, ts["task_goal"])
    step_scores = [d.get("arg_score", 0.0) for d in ts["expected"] if d["op"] != "cancel"]
    aa["final_mean"] = aa["mean"]
    aa["operation_states"] = ts["expected"]
    values = step_scores + [v["score"] for v in aa["goals"].values() if v["scored"]]
    aa["mean"] = round(sum(values) / len(values), 4) if values else None
    dl = deliveries(ep, run, ts["task_goal"])
    tl = timing_layer(ep, run, ts["observed_ops"])
    strict = tool_selection(ep, run, enforce_deadlines=True)
    tl['deadline_tool_selection_f1'] = strict['f1']
    tl['matched_operations'] = ts['tp']
    tl['on_time_operations'] = sum(d.get('on_time', False) for d in ts['expected'])
    tl['late_operations'] = sum(d['matched'] and not d.get('on_time', False) for d in ts['expected'])
    deliveries_ok = all(v.get("delivered", True) and v.get("problem_reported", True) is not False for v in dl.values())
    hard = {
        "tool_selection_exact": ts["fn"] == 0 and ts["fp"] == 0,
        "args_exact": all(v.get("exact", True) and v.get("status_ok", True) for v in aa["goals"].values())
                      and all(d.get("args_exact", False) for d in ts["expected"] if d["op"] != "cancel"),
        "deliveries_ok": deliveries_ok,
        "all_turns_played": not run.get("unplayed_user_turns"),
    }
    return {"tool_selection": ts, "arg_accuracy": aa, "deliveries": dl, "timing": tl, "hard": hard}
