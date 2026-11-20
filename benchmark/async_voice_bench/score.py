import statistics
from collections import defaultdict
from .semantic import semantic_delivery_success


PASS_AT_1_DEFINITION = 'tool_selection_exact_and_args_exact_v1'


def tool_pass_checks(prog):
    return {key: bool(prog['hard'][key]) for key in ('tool_selection_exact', 'args_exact')}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 4) if xs else None


def _n01(x):
    return None if x is None else round((x - 1) / 4, 4)


def naturalness_breakdown(prog, epj):
    """Group turn scores by dispatch, waiting, and other turns; include applicable delivery."""
    tn = (epj or {}).get("turn_naturalness") or {}
    tt = prog["timing"]["turns"]
    groups = defaultdict(list)
    for tid, sc in tn.items():
        if tid not in tt or not isinstance(sc, (int, float)):
            continue
        v = tt[tid]
        groups["dispatch" if v["dispatch"] else ("wait" if v["busy"] else "other")].append(sc)
    phases = {k: _mean(v) for k, v in groups.items()}
    d = (epj or {}).get("delivery_naturalness")
    if any("delivered" in v for v in prog["deliveries"].values()) and d is None:
        d = 1  # A required but absent delivery cannot disappear from the denominator.
    if isinstance(d, (int, float)):
        phases["delivery"] = float(d)
    # Weight turns equally; phase averages are diagnostic only.
    core = list(tn.values()) + ([d] if d is not None else [])
    overall = _mean(core)
    return {"phases_1to5": phases, "overall_1to5": overall, "turns_1to5": tn,
            "busy_turns": [t for t, v in tt.items() if v["busy"] or v["dispatch"]]}


def episode_scores(ep, prog, jud):
    cps = dict(jud["checkpoints"])
    epj = dict(jud.get("episode") or {})
    turn_scores = dict(epj.get('turn_naturalness') or {})
    for turn in ep['turns']:
        if turn['turn_id'] not in prog['timing']['turns']:
            turn_scores[turn['turn_id']] = 1
    epj['turn_naturalness'] = turn_scores
    q = [(v.get("quality", 1) - 1) / 4 for v in cps.values()]
    if epj.get("result_faithfulness"):
        q.append((epj["result_faithfulness"] - 1) / 4)
    lat_elig = [v for v in cps.values() if v.get("latency_eligible", True)]
    lats = [v["followup_latency_s"] for v in lat_elig if v.get("answered") and v.get("followup_latency_s") is not None]
    ok = lambda v: v.get("answered") and v.get("grounded", True) and not v.get("fabrication")
    judge_ok = all(ok(v) for v in cps.values()) and not epj.get("task_confusion") and not epj.get("fabricated_progress")
    hard = dict(prog['hard'])
    semantic = jud.get('semantic')
    if semantic is not None:
        hard['deliveries_ok'] = semantic_delivery_success(semantic)
        hard['results_faithful'] = semantic['results_faithful']
    elif jud['judge'].startswith('llm:'):
        raise ValueError('LLM scoring requires validated semantic delivery judgments')
    # Interaction success requires result faithfulness of at least 4/5.
    hard['result_faithfulness_acceptable'] = epj.get('result_faithfulness', 1) >= 4
    fail_reasons = [k for k, v in hard.items() if not v]
    fail_reasons += [f"checkpoint:{k}" for k, v in cps.items() if not ok(v)]
    fail_reasons += [k for k in ("task_confusion", "fabricated_progress") if epj.get(k)]
    nb = naturalness_breakdown(prog, epj)
    if semantic is not None and any(row['delivered'] is False for row in semantic['goals'].values()):
        nb = naturalness_breakdown(prog, {**epj, 'delivery_naturalness': 1})
    tl = prog["timing"]
    pass_checks = tool_pass_checks(prog)
    return {
        "episode_id": ep["episode_id"], "lang": ep["lang"], "split": ep["split"], "template_id": ep["template_id"],
        "scenario_tags": ep["scenario_tags"],
        "tool_selection": prog["tool_selection"]["f1"],
        "arg_accuracy": prog["arg_accuracy"]["mean"],
        "pass": all(pass_checks.values()),
        "pass_definition": PASS_AT_1_DEFINITION,
        "pass_checks": pass_checks,
        "pass_tool_args_only": all(pass_checks.values()),
        "interaction_success": all(hard.values()) and judge_ok,
        "response_quality": round(sum(q) / len(q), 4) if q else None,
        "naturalness": _n01(nb["overall_1to5"]),
        "naturalness_phases": {k: _n01(v) for k, v in nb["phases_1to5"].items()},
        "turn_naturalness_1to5": nb["turns_1to5"], "busy_turns": nb["busy_turns"],
        "followup_latencies": lats,
        "followup_answered": sum(1 for v in lat_elig if v.get("answered")),
        "followup_total": len(lat_elig),
        "fail_reasons": [key for key, value in pass_checks.items() if not value],
        "interaction_fail_reasons": fail_reasons,
        "hard_checks": hard,
        "delivery_judge": 'semantic' if semantic is not None else 'heuristic_probe_only',
        "diagnostics": {k: tl[k] for k in ("talk_over", "interrupted_segments", "dropped_planned_speech",
                                           "dispatch_silence_s", "status_queries", "rejected_calls")}
        | {"stalling": epj.get("stalling"),
           "deadline_tool_selection_f1": tl['deadline_tool_selection_f1'],
           "matched_operations": tl['matched_operations'],
           "on_time_operations": tl['on_time_operations'],
           "late_operations": tl['late_operations'],
           "overlap_seconds": tl.get('overlap_seconds', 0),
           "unprompted_repeats": sum(d.get("unprompted_repeats", 0) for d in prog["deliveries"].values()),
           "delivery_delays_s": {g: d.get("delay_s") for g, d in prog["deliveries"].items() if "delay_s" in d}},
    }


def _p(vals, q):
    if not vals:
        return None
    v = sorted(vals)
    return round(v[min(len(v) - 1, int(round(q * (len(v) - 1))))], 2)


COLS = ["n", "ToolSel", "ArgAcc", "Pass@1", "RespQual", "Natural", "FU-Lat p50", "FU-Lat p90", "FU answered"]
NAT_COLS = ["Nat-dispatch", "Nat-wait", "Nat-delivery", "DispatchSilence p50", "TalkOver/ep"]


def aggregate(rows):
    if any(r.get('pass_definition') != PASS_AT_1_DEFINITION for r in rows):
        raise ValueError('Recompute rows with the current Pass@1 definition before aggregation')
    def agg(rs):
        if not rs:
            return None
        lats = [x for r in rs for x in r["followup_latencies"]]
        ds = [x for r in rs for x in r["diagnostics"]["dispatch_silence_s"]]
        ans = sum(r["followup_answered"] for r in rs)
        tot = sum(r["followup_total"] for r in rs)
        mean = lambda k: _mean([r[k] for r in rs])
        ph = lambda k: _mean([r["naturalness_phases"].get(k) for r in rs])
        return {"n": len(rs), "ToolSel": mean("tool_selection"), "ArgAcc": mean("arg_accuracy"),
                "Pass@1": round(sum(r["pass"] for r in rs) / len(rs), 3), "RespQual": mean("response_quality"),
                "Natural": mean("naturalness"), "FU-Lat p50": _p(lats, 0.5), "FU-Lat p90": _p(lats, 0.9),
                "FU answered": f"{ans}/{tot}",
                "Nat-dispatch": ph("dispatch"), "Nat-wait": ph("wait"), "Nat-delivery": ph("delivery"),
                "DispatchSilence p50": _p(ds, 0.5),
                "TalkOver/ep": round(sum(r["diagnostics"]["talk_over"] for r in rs) / len(rs), 2),
                "InteractionSuccess": round(sum(r["interaction_success"] for r in rs) / len(rs), 3)}

    out = {"all": agg(rows)}
    for key in ("lang",):
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        for g, rs in sorted(groups.items()):
            out[f"{key}={g}"] = agg(rs)
    tags = defaultdict(list)
    for r in rows:
        for t in r["scenario_tags"]:
            tags[t].append(r)
    for t, rs in sorted(tags.items()):
        out[f"scenario={t}"] = agg(rs)
    return out


def to_markdown(system, judge_name, table):
    def tbl(cols, keys):
        lines = ["| slice | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
        for k in keys:
            v = table[k]
            if v:
                lines.append(f"| {k} | " + " | ".join("" if v.get(c) is None else str(v.get(c)) for c in cols) + " |")
        return lines
    keys = list(table)
    head = [f"### {system}  (judge: {judge_name})", "", "**Headline**", ""]
    nat = ["", "**Naturalness breakdown (0-1) + timing evidence**", ""]
    return "\n".join(head + tbl(COLS, keys) + nat +
                     tbl(NAT_COLS + ["InteractionSuccess"], [k for k in keys if not k.startswith("scenario=")]))
