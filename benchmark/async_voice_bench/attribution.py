"""Match tasks to evaluator-only goals and check argument constraints."""
from .textutil import norm
from .tools_catalog import canon


class _Missing:
    def __repr__(self):
        return "<missing>"


_MISSING = _Missing()


def value_ok(actual, spec, aliases) -> bool:
    """Constraint language:
       plain value             -> equal after alias canonicalisation / normalisation
       {"$oneof": [...]}       -> equal to any
       {"$contains": [...]}    -> list includes all
       {"$excludes": [...]}    -> list includes none
       {"$range": [lo, hi]}    -> numeric range (inclusive)
       {"$absent": true}       -> key must be absent
       {"$any": true}          -> anything, including absent
       {"$all": [spec, ...]}   -> conjunction
       {"$optional": spec}     -> absent, or satisfies spec
    """
    if isinstance(spec, dict):
        if spec.get("$any"):
            return True
        if "$absent" in spec:
            return actual is _MISSING
        if actual is _MISSING:
            return "$optional" in spec
        if "$optional" in spec:
            return value_ok(actual, spec["$optional"], aliases)
        if "$all" in spec:
            return all(value_ok(actual, s, aliases) for s in spec["$all"])
        if "$oneof" in spec:
            return any(value_ok(actual, s, aliases) for s in spec["$oneof"])
        if "$contains" in spec:
            return isinstance(actual, list) and all(
                any(norm(canon(a, aliases)) == norm(canon(x, aliases)) for a in actual) for x in spec["$contains"])
        if "$excludes" in spec:
            return isinstance(actual, list) and not any(
                norm(canon(a, aliases)) == norm(canon(x, aliases)) for a in actual for x in spec["$excludes"])
        if "$range" in spec:
            lo, hi = spec["$range"]
            return isinstance(actual, (int, float)) and not isinstance(actual, bool) and lo <= actual <= hi
        return False
    if actual is _MISSING:
        return False
    if isinstance(spec, bool) or isinstance(actual, bool):
        return actual is spec
    if isinstance(spec, (int, float)) and isinstance(actual, (int, float)):
        return abs(actual - spec) < 1e-6
    if isinstance(spec, list):
        if not isinstance(actual, list):
            return False
        return sorted(norm(canon(x, aliases)) for x in actual) == sorted(norm(canon(x, aliases)) for x in spec)
    return norm(canon(actual, aliases)) == norm(canon(spec, aliases))


def check_args(args: dict, constraints: dict, aliases, allowed_extra=()):
    """Return correct count, total count, and errors; undeclared extra arguments count as errors."""
    errors, ok = [], 0
    for k, spec in constraints.items():
        actual = args.get(k, _MISSING)
        if value_ok(actual, spec, aliases):
            ok += 1
        else:
            errors.append({"slot": k, "expected": spec, "actual": None if actual is _MISSING else actual})
    extra = [k for k in args if k not in constraints and k not in allowed_extra]
    for k in extra:
        errors.append({"slot": k, "expected": "<not requested>", "actual": args[k]})
    return ok, len(constraints) + len(extra), errors


def attribute_task(tool, versions, goals, aliases):
    best, best_score = None, -1
    for g in goals:
        m = g["match"]
        if m["tool"] != tool:
            continue
        keys = m.get("args", {})
        for ver in versions:
            if all(value_ok(ver["args"].get(k, _MISSING), v, aliases) for k, v in keys.items()):
                if len(keys) > best_score:
                    best, best_score = g["goal_id"], len(keys)
                break
    return best
