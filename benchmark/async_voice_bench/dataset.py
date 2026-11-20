import glob
import json
import os

from .tools_catalog import TOOLS, validate_materials

TRIGGERS = {"at", "after_turn", "after_agent_idle", "on_event", "after_agent_mention"}
CP_TYPES = {"knowledge_answer", "status_answer", "constraint_update", "chitchat_no_action", "clarification",
            "partial_answer", "grounded_ack", "grounded_answer", "selective_cancel", "silence_request", "resume",
            "problem_ack", "no_change", "dependent_request"}


def load_episodes(root):
    eps = []
    for p in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
        eps.append(json.load(open(p, encoding="utf-8")))
    return eps


def validate(ep):
    errs = []
    for k in ("episode_id", "template_id", "split", "lang", "scenario_tags", "user_goal", "environment", "turns", "goals"):
        if k not in ep:
            errs.append(f"missing {k}")
    if errs:
        return errs
    tids = [t["turn_id"] for t in ep["turns"]]
    if len(set(tids)) != len(tids):
        errs.append("duplicate turn ids")
    gids = {g["goal_id"] for g in ep["goals"]}
    for t in ep["turns"]:
        trg = t.get("trigger", {})
        if trg.get("type") not in TRIGGERS:
            errs.append(f"{t['turn_id']}: bad trigger type {trg.get('type')}")
        if trg.get("ref") and trg["ref"] not in tids:
            errs.append(f"{t['turn_id']}: trigger ref {trg['ref']} unknown")
        if trg.get("goal") and trg["goal"] not in gids:
            errs.append(f"{t['turn_id']}: trigger goal unknown")
    for g in ep["goals"]:
        if g["match"]["tool"] not in TOOLS:
            errs.append(f"{g['goal_id']}: unknown tool")
            continue
        for o in g.get("ops", []):
            if o["after"] not in tids:
                errs.append(f"{g['goal_id']}: op after unknown turn {o['after']}")
            if o["op"] not in ("start", "modify", "cancel", "retry"):
                errs.append(f"{g['goal_id']}: bad op {o['op']}")
            if o["op"] != "cancel" and not isinstance(o.get("constraints"), dict):
                errs.append(f"{g['goal_id']}: missing operation-state constraints after {o['after']}")
        for k in g["final"].get("constraints", {}):
            if k not in TOOLS[g["match"]["tool"]]["params"]:
                errs.append(f"{g['goal_id']}: constraint on unknown param {k}")
        d = g.get("delivery")
        if d and d.get("required") and not all(isinstance(x, list) for x in d.get("key_facts", [])):
            errs.append(f"{g['goal_id']}: key_facts must be a list of alternative-lists")
    for c in ep.get("checkpoints", []):
        if c["turn_id"] not in tids:
            errs.append(f"checkpoint {c['cp_id']} unknown turn")
        if c["type"] not in CP_TYPES:
            errs.append(f"checkpoint {c['cp_id']} unknown type {c['type']}")
        kw = c.get("judge_hints", {}).get("answer_keywords", [])
        if not all(isinstance(x, list) for x in kw):
            errs.append(f"checkpoint {c['cp_id']}: answer_keywords must be a list of alternative-lists")
    for tool, beh in ep["environment"].get("tool_behaviors", {}).items():
        if tool not in TOOLS:
            errs.append(f"behavior for unknown tool {tool}")
        else:
            error = validate_materials(tool, ep['environment'].get('materials', {}))
            if error:
                errs.append(f"configured tool missing materials: {tool}")
        for anc in (beh.get("release_not_before"), beh.get("amend_policy", {}).get("release_not_before")):
            if anc and anc["turn"] not in tids:
                errs.append(f"{tool}: release anchor turn {anc['turn']} unknown")
    op_turns = {o["after"] for g in ep["goals"] for o in g.get("ops", [])}
    if not [t for t in tids if t not in op_turns]:
        errs.append("design rule: at least one user turn must NOT require any backend action")
    return errs
