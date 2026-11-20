import argparse
import json
import os
import sys

from .dataset import load_episodes, validate
from .grader import programmatic
from .judge import judge_episode, OpenAIJudge, render_transcript
from .score import episode_scores, aggregate, to_markdown
from .protocol import load_registered_runs, write_json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "episodes")


def cmd_validate(a):
    eps = load_episodes(a.data)
    bad = 0
    for e in eps:
        errs = validate(e)
        if errs:
            bad += 1
            print(f"[FAIL] {e['episode_id']}: {errs}")
    n = {s: sum(1 for e in eps if e["split"] == s) for s in sorted({e['split'] for e in eps})}
    langs = {l: sum(1 for e in eps if e["lang"] == l) for l in ("en", "zh")}
    print(f"{len(eps)} episodes  splits={n}  langs={langs}  templates={len({e['template_id'] for e in eps})}")
    sys.exit(1 if bad else 0)


def _evaluate_dir(runs_dir, eps, judge, out=None):
    manifest, runs, pending = load_registered_runs(runs_dir, eps)
    rows, system, full = [], manifest['system_id'], {}
    status = {'expected': len(manifest['episode_ids']),
              'pending': pending, 'judge_errors': {}, 'scored': 0, 'complete': False}
    if out:
        write_json(os.path.join(out, 'evaluation_status.json'), status)
    if pending:
        raise ValueError(f'Run set incomplete: {pending}; no headline score produced')
    for eid in manifest['episode_ids']:
        run = runs[eid]
        ep = eps[eid]
        try:
            prog = programmatic(ep, run)
            jud = judge_episode(ep, run, judge, prog['timing'])
            sc = episode_scores(ep, prog, jud)
            if run['execution_status'] == 'model_failure':
                sc['interaction_success'] = False
                sc['interaction_fail_reasons'].append('model_execution_failure')
        except Exception as exc:
            status['judge_errors'][eid] = {'type': type(exc).__name__}
            if out:
                write_json(os.path.join(out, 'evaluation_status.json'), status)
            raise
        rows.append(sc)
        full[eid] = {"scores": sc, "programmatic": prog, "judge": jud}
        if out:
            write_json(os.path.join(out, 'episodes', eid + '.json'), full[eid])
        status['scored'] += 1
        if out:
            write_json(os.path.join(out, 'evaluation_status.json'), status)
    status['complete'] = True
    if out:
        write_json(os.path.join(out, 'evaluation_status.json'), status)
    return rows, system, full


def cmd_evaluate(a):
    eps = {e["episode_id"]: e for e in load_episodes(a.data)}
    judge = OpenAIJudge(a.judge_model, cache_path=os.path.join(a.out, "judge_cache.jsonl"))
    rows, system, _ = _evaluate_dir(a.runs, eps, judge, a.out)
    table = aggregate(rows)
    manifest = json.load(open(os.path.join(a.runs, 'manifest.json')))
    write_json(os.path.join(a.out, "summary.json"), {"system": system, "judge": judge.name,
        "judge_config": getattr(judge, 'config', None), "manifest": manifest,
        "reportable": manifest['timing_source'] == 'external_realtime_recording',
        "table": table, "rows": rows})
    md = to_markdown(system, judge.name, table)
    open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8").write(md + "\n")
    print(md)
    print()
    for r in rows:
        ph = " ".join(f"{k}={v}" for k, v in r["naturalness_phases"].items())
        print(f"  {r['episode_id']:<42} pass={int(r['pass'])} TS={r['tool_selection']:.2f} "
              f"AA={'-' if r['arg_accuracy'] is None else round(r['arg_accuracy'], 2)} Nat={r['naturalness']} [{ph}] fail={r['fail_reasons']}")


def cmd_render(a):
    run = json.load(open(a.run, encoding="utf-8"))
    print(render_transcript(run, with_truth=not a.no_truth)[0])


def main():
    p = argparse.ArgumentParser(prog="asyncvoicebench")
    sp = p.add_subparsers(dest="cmd", required=True)

    v = sp.add_parser("validate"); v.add_argument("--data", default=DATA); v.set_defaults(f=cmd_validate)
    e = sp.add_parser("evaluate"); e.add_argument("--data", default=DATA); e.add_argument("--runs", required=True)
    e.add_argument("--out", required=True); e.add_argument("--judge-model", default="gpt-5.2-2025-12-11")
    e.set_defaults(f=cmd_evaluate)
    rd = sp.add_parser("render"); rd.add_argument("run"); rd.add_argument("--no-truth", action="store_true"); rd.set_defaults(f=cmd_render)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
