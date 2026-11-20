import argparse
from collections import Counter
from pathlib import Path
import statistics

from .dataset import load_episodes
from .grader import programmatic
from .protocol import load_registered_runs, write_json
from .score import PASS_AT_1_DEFINITION, tool_pass_checks


ROOT = Path(__file__).resolve().parents[1]


def summarize(runs_directory, output):
    episodes = {ep['episode_id']: ep for ep in load_episodes(str(ROOT / 'data/episodes'))}
    manifest, runs, pending = load_registered_runs(runs_directory, episodes)
    rows = []
    for eid in manifest['episode_ids']:
        if eid not in runs:
            continue
        run = runs[eid]
        ep = episodes[eid]
        checked = programmatic(ep, run)
        hard = {key: checked['hard'][key] for key in ('tool_selection_exact', 'args_exact', 'all_turns_played')}
        hard['model_execution_ok'] = run['execution_status'] != 'model_failure'
        pass_checks = tool_pass_checks(checked)
        rows.append(dict(episode_id=eid, split=ep['split'], template_id=ep['template_id'],
            tool_pass=all(pass_checks.values()), pass_definition=PASS_AT_1_DEFINITION,
            pass_failure_reasons=[key for key, value in pass_checks.items() if not value],
            execution_status=run['execution_status'], tool_selection=checked['tool_selection']['f1'],
            arg_accuracy=checked['arg_accuracy']['mean'], execution_success=all(hard.values()),
            failure_reasons=[key for key, value in hard.items() if not value],
            rejected_calls=checked['timing']['rejected_calls'],
            timing=checked['timing'],
            operations=checked['tool_selection'], arguments=checked['arg_accuracy']))

    def aggregate(selected):
        def mean(key):
            values = [r[key] for r in selected if r[key] is not None]
            return round(statistics.mean(values), 4) if values else None
        return dict(n=len(selected), ToolSel=mean('tool_selection'), ArgAcc=mean('arg_accuracy'),
                    **{'Pass@1': mean('tool_pass')},
                    execution_success_rate=mean('execution_success'),
                    execution_successes=sum(r['execution_success'] for r in selected),
                    rejected_calls=sum(r['rejected_calls'] for r in selected),
                    matched_operations=sum(r['timing']['matched_operations'] for r in selected),
                    on_time_operations=sum(r['timing']['on_time_operations'] for r in selected),
                    late_operations=sum(r['timing']['late_operations'] for r in selected))

    report = dict(system=manifest['system_id'], evaluation='programmatic_only',
        pass_definition=PASS_AT_1_DEFINITION,
        llm_judge_called=False, expected=len(manifest['episode_ids']), scored=len(rows),
        complete=not pending, pending=pending,
        pending_counts=dict(Counter(pending.values())),
        scope_note='Provisional averages use scored cases only until complete. Pass@1 requires exact tool selection and arguments. Execution success additionally checks all user turns were played and no model execution failure occurred.',
        withheld_metrics=['RespQual', 'Natural', 'FU-Ans', 'FU-Lat'],
        table={'all': aggregate(rows)}, rows=rows)
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.runs, args.out)
    print(f"Programmatic only: {report['scored']}/{report['expected']}; complete={report['complete']}")
    print(report['table']['all'])


if __name__ == '__main__':
    main()
