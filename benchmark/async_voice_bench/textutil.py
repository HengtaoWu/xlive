import re
import unicodedata


def norm(s) -> str:
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    s = re.sub(r"[\s,，、'’\"“”$¥￥]", "", s)
    return s


def match_groups(text: str, groups, min_groups: int = 1) -> bool:
    """Match at least min_groups groups, with any alternative satisfying each group."""
    if not groups:
        return False
    t = norm(text)
    hit = sum(1 for g in groups if any(norm(a) in t for a in g))
    return hit >= min_groups
