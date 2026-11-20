"""Domain tool schemas and deterministic executors using only arguments and session materials."""
from datetime import date, time
from contextlib import closing
import math
import re
import sqlite3
from .textutil import norm

# Parameter types: str, int, num, bool, date (YYYY-MM-DD), time (HH:mm), list[str], enum.
TOOLS = {
    "search_flights": {
        "description": "Search one-way flights. Cities by name. Returns matching flights sorted by price.",
        "params": {"origin": "str", "destination": "str", "date": "date", "max_price": "num",
                   "nonstop_only": "bool", "depart_after": "time"},
        "required": ["origin", "destination", "date"],
    },
    "search_hotels": {
        "description": "Search hotels in a city for a stay. area is an optional named neighborhood matched literally, not a qualitative preference such as near a station. Station proximity uses max_walk_min_to_station (integer minutes); if no limit is given, omit it and compare the returned walk_min values. Optional max_price_per_night is a nightly price cap.",
        "params": {"city": "str", "check_in": "date", "check_out": "date", "area": "str",
                   "max_price_per_night": "num", "max_walk_min_to_station": "int"},
        "required": ["city", "check_in", "check_out"],
    },
    "compare_products": {
        "description": "Compare named products side by side on chosen aspects (e.g. price, cpu, ram_gb, storage_gb, weight_kg, battery_h, display).",
        "params": {"products": "list[str]", "aspects": "list[str]"},
        "required": ["products"],
    },
    "search_products": {
        "description": "Search a product category matched by name or a documented alias. For a device-specific accessory, include its device model in category (the device model followed by case, for example); a generic phone case category cannot specify compatibility. features are exact required tags such as quiet, wireless, compact, backlit, mechanical, rugged. Do not put device names or descriptive sentences in features. Return all matching items.",
        "params": {"category": "str", "features": "list[str]", "max_price": "num"},
        "required": ["category"],
    },
    "search_restaurants": {
        "description": "Find restaurants with a table for the party. Optional cuisine and accessibility requirement. Optional time is an earliest HH:mm local time; omit it when no precise time is specified and return available times for the user to choose.",
        "params": {"area": "str", "party_size": "int", "date": "date", "time": "time",
                   "wheelchair_accessible": "bool", "cuisine": "str"},
        "required": ["area", "party_size", "date"],
    },
    "analyze_sales": {
        "description": "Aggregate sales for a month (YYYY-MM). metric: revenue|order_count. group_by: region|channel.",
        "params": {"month": "str", "group_by": "enum:region|channel", "metric": "enum:revenue|order_count",
                   "exclude_refunded": "bool"},
        "required": ["month", "group_by", "metric"],
    },
    "compare_quotes": {
        "description": "Collect and compare vendor quotes against a budget. Install fees may arrive later than base prices.",
        "params": {"vendors": "list[str]", "include_install_fee": "bool", "budget": "num"},
        "required": ["vendors"],
    },
    "summarize_document": {
        "description": "Summarize a document from the user's files into meeting-minute style points.",
        "params": {"file": "str", "focus": "str"},
        "required": ["file"],
    },
    "generate_report": {
        "description": "Draft a multi-section report. Long-running; exposes progress stages.",
        "params": {"topic": "str", "period": "str", "sections": "list[str]"},
        "required": ["topic"],
    },
    "research_sources": {
        "description": "Search the session's frozen reference corpus by topic. Returns relevant source records and any available summaries; this is not live web browsing. An empty result means the corpus has no matching reference, not that the requested fact is false.",
        "params": {"topic": "str", "max_items": "int"},
        "required": ["topic"],
    },
    "triage_logs": {
        "description": "Scan a service's error logs in a time window and group errors. time_window: last_1h|last_night|last_24h|last_7d.",
        "params": {"service": "str", "time_window": "enum:last_1h|last_night|last_24h|last_7d",
                   "min_level": "enum:warn|error"},
        "required": ["service", "time_window"],
    },
    "find_meeting_slots": {
        "description": "Find common free windows for attendees on a date (include the user as 'me' if they attend). part_of_day: morning|afternoon|any.",
        "params": {"attendees": "list[str]", "date": "date", "part_of_day": "enum:morning|afternoon|any",
                   "duration_min": "int"},
        "required": ["attendees", "date", "duration_min"],
    },
    "search_rentals": {
        "description": "Search apartments for rent in a city. Optional max monthly rent, bedrooms, pets allowed, furnished, max commute minutes to the user's workplace.",
        "params": {"city": "str", "max_rent": "num", "bedrooms": "int", "pets_allowed": "bool", "furnished": "bool",
                   "max_commute_min": "int"},
        "required": ["city"],
    },
    "track_order": {
        "description": "Look up the shipping status of one order by order id.",
        "params": {"order_id": "str"},
        "required": ["order_id"],
    },
    "find_recipes": {
        "description": "Find recipes that use the given ingredients. Optional diet (vegetarian|vegan|gluten_free|nut_free), max total minutes, servings.",
        "params": {"ingredients": "list[str]", "diet": "enum:vegetarian|vegan|gluten_free|nut_free", "max_time_min": "int",
                   "servings": "int"},
        "required": ["ingredients"],
    },
    "search_events": {
        "description": "Search events in a city on a date. Optional category is a case-insensitive literal substring of an event's category, such as baseball or concert; use one category label only, never append team names, venues or the full request. Omit category when unsure. Filter returned events by the user's named teams or performers when presenting results. Optional max_price caps ticket price.",
        "params": {"city": "str", "date": "date", "category": "str", "max_price": "num"},
        "required": ["city", "date"],
    },
}


def canon(value, aliases: dict):
    if not isinstance(value, str):
        return value
    v = norm(value)
    for canonical, alts in (aliases or {}).items():
        if v == norm(canonical) or v in {norm(a) for a in alts}:
            return canonical
    return value


def validate_args(tool: str, args: dict):
    if not isinstance(tool, str):
        return "tool must be a string"
    spec = TOOLS.get(tool)
    if spec is None:
        return f"unknown tool '{tool}'"
    if not isinstance(args, dict):
        return "args must be an object"
    for r in spec["required"]:
        if r not in args or args[r] in (None, "", []):
            return f"missing required argument '{r}'"
    for k, v in args.items():
        t = spec["params"].get(k)
        if t is None:
            return f"unknown argument '{k}' for {tool}"
        if t.startswith("enum:") and v not in t[5:].split("|"):
            return f"argument '{k}' must be one of {t[5:]}"
        if t == "bool" and not isinstance(v, bool):
            return f"argument '{k}' must be boolean"
        if t in ("int", "num") and (isinstance(v, bool) or not isinstance(v, (int, float))):
            return f"argument '{k}' must be a number"
        if t in ("int", "num") and not math.isfinite(v):
            return f"argument '{k}' must be finite"
        if t == "int" and type(v) is not int:
            return f"argument '{k}' must be an integer"
        if t in ("str", "date", "time") and (not isinstance(v, str) or not v.strip()):
            return f"argument '{k}' must be non-empty text"
        if t == "date":
            try:
                parsed = date.fromisoformat(v)
            except ValueError:
                return f"argument '{k}' must be an ISO date"
            if parsed.isoformat() != v:
                return f"argument '{k}' must use YYYY-MM-DD"
        if t == 'time':
            try:
                parsed = time.fromisoformat(v)
            except ValueError:
                return f"argument '{k}' must use HH:mm local time"
            if parsed.tzinfo is not None or parsed.isoformat(timespec='minutes') != v:
                return f"argument '{k}' must use HH:mm local time"
        if t.startswith("list") and not isinstance(v, list):
            return f"argument '{k}' must be a list"
        if t == "list[str]" and any(not isinstance(x, str) or not x.strip() for x in v):
            return f"argument '{k}' must contain non-empty strings"
    if tool == "research_sources" and args.get("max_items", 5) <= 0:
        return "argument 'max_items' must be a positive integer"
    return None


def _eq(a, b, aliases):
    return norm(canon(a, aliases)) == norm(canon(b, aliases))


def ex_search_flights(a, m):
    al = m.get("aliases", {})
    out = []
    for f in m["flights"]:
        if not (_eq(f["origin"], a["origin"], al) and _eq(f["destination"], a["destination"], al)
                and f["date"] == a["date"]):
            continue
        if "max_price" in a and f["price"] > a["max_price"]:
            continue
        if a.get("nonstop_only") and f["stops"] > 0:
            continue
        if a.get("depart_after") and f["depart"] < a["depart_after"]:
            continue
        out.append(f)
    out.sort(key=lambda f: f["price"])
    return {"count": len(out), "flights": out, "currency": m.get("currency", "USD")}


def ex_search_hotels(a, m):
    al = m.get("aliases", {})
    out = []
    for h in m["hotels"]:
        if not _eq(h["city"], a["city"], al):
            continue
        if a.get("area") and not _eq(h["area"], a["area"], al):
            continue
        if "max_price_per_night" in a and h["price_per_night"] > a["max_price_per_night"]:
            continue
        if "max_walk_min_to_station" in a and h["walk_min"] > a["max_walk_min_to_station"]:
            continue
        out.append(h)
    out.sort(key=lambda h: (-h["rating"], h["price_per_night"]))
    return {"count": len(out), "hotels": out, "check_in": a["check_in"], "check_out": a["check_out"]}


def ex_compare_products(a, m):
    al = m.get("aliases", {})
    rows, missing = [], []
    for name in a["products"]:
        cn = canon(name, al)
        p = next((p for p in m["products"] if norm(p["name"]) == norm(cn)), None)
        (rows.append(p) if p else missing.append(name))
    if missing:
        return {"__error__": f"product(s) not found: {', '.join(missing)}"}
    aspects = [canon(x, al) for x in a.get("aspects") or []] or [k for k in rows[0] if k != "name"]
    table = [{"name": r["name"], **{k: r.get(k) for k in aspects}} for r in rows]
    cheapest = min(rows, key=lambda r: r["price"])["name"] if all("price" in r for r in rows) else None
    return {"table": table, "cheapest": cheapest}


def ex_search_products(a, m):
    aliases = m.get('aliases', {})
    feats = {norm(canon(f, aliases)) for f in a.get('features') or []}
    out = [p for p in m["catalog"] if _eq(p['category'], a['category'], aliases)
           and feats <= {norm(canon(f, aliases)) for f in p['features']}
           and p["price"] <= a.get("max_price", 1e12)]
    out.sort(key=lambda p: (-p["rating"], p["price"]))
    return {"count": len(out), "items": out}


def ex_search_restaurants(a, m):
    al = m.get("aliases", {})
    out = []
    for r in m["restaurants"]:
        if not _eq(r["area"], a["area"], al):
            continue
        if r["max_party"] < a["party_size"]:
            continue
        if a.get("wheelchair_accessible") and not r["wheelchair_accessible"]:
            continue
        if a.get("cuisine") and norm(a["cuisine"]) not in norm(r["cuisine"]):
            continue
        slots = r["slots"].get(a["date"], [])
        if a.get("time"):
            slots = [s for s in slots if s >= a["time"]]
        if not slots:
            continue
        out.append({**{k: r[k] for k in ("name", "area", "cuisine", "price_pp", "rating", "wheelchair_accessible")},
                    "available_times": slots})
    out.sort(key=lambda r: -r["rating"])
    return {"count": len(out), "restaurants": out}


def ex_analyze_sales(a, m):
    key = a["group_by"]
    agg = {}
    for row in m["sales_rows"]:
        if row["month"] != a["month"]:
            continue
        if a.get("exclude_refunded") and row["status"] == "refunded":
            continue
        g = row[key]
        val = row["revenue"] if a["metric"] == "revenue" else row["orders"]
        agg[g] = agg.get(g, 0) + val
    if not agg:
        return {"__error__": f"no sales rows for month {a['month']}"}
    ranking = sorted(agg.items(), key=lambda kv: -kv[1])
    return {"metric": a["metric"], "group_by": key, "month": a["month"],
            "exclude_refunded": bool(a.get("exclude_refunded")),
            "ranking": [{"group": g, "value": round(v, 2)} for g, v in ranking]}


def _quote_rows(a, m, t_rel, final):
    rows = []
    for name in a["vendors"]:
        v = next((v for v in m["vendors"] if norm(v["name"]) == norm(canon(name, m.get("aliases", {})))), None)
        if v is None:
            rows.append({"vendor": name, "error": "unknown vendor"})
            continue
        fee_known = final or v.get("install_fee_ready_s", 0) <= t_rel
        fee = v["install_fee"] if fee_known else None
        inc = a.get("include_install_fee", True)
        row = {"vendor": v["name"], "base_price": v["base_price"], "install_fee": fee if inc else "excluded"}
        if inc:
            row["total"] = v["base_price"] + fee if fee is not None else None
        else:
            row["total"] = v["base_price"]
        rows.append(row)
    return rows


def ex_compare_quotes(a, m, t_rel=None, final=True):
    rows = _quote_rows(a, m, t_rel or 0, final)
    known = [r for r in rows if r.get("total") is not None]
    res = {"quotes": rows, "budget": a.get("budget")}
    if len(known) == len(rows):
        best = min(rows, key=lambda r: r["total"])
        res["cheapest"] = best["vendor"]
        if a.get("budget") is not None:
            res["within_budget"] = [r["vendor"] for r in rows if r["total"] <= a["budget"]]
    else:
        res["incomplete"] = [r["vendor"] for r in rows if r.get("total") is None]
    return res


def ex_summarize_document(a, m):
    al = m.get("aliases", {})
    f = m["files"].get(a["file"]) or next((v for k, v in m["files"].items()
                                          if norm(k) == norm(canon(a["file"], al))), None)
    if f is None:
        return {"__needs_input__": {"reason": "file_not_found", "detail": f"No file named '{a['file']}' in the user's files."}}
    if not f.get("readable", True):
        return {"__needs_input__": {"reason": "unreadable_attachment", "detail": f["error"]}}
    return {"file": a["file"], "points": f["points"]}


def ex_generate_report(a, m):
    r = m["report"]
    return {"title": r["title"], "sections": r["sections"], "headline": r["headline"]}


def ex_research_sources(a, m):
    sources = m["sources"]
    stop_words = set("a an and are as at be by can could do does for from how i in is it of on or that the this to use what whether which with you".split())
    terms = sorted(set(re.findall(r"\w+", a["topic"].casefold())) - stop_words)
    rows = []
    if terms and sources:
        with closing(sqlite3.connect(":memory:")) as db:
            db.execute("CREATE VIRTUAL TABLE refs USING fts5(content, tokenize='porter unicode61')")
            db.executemany("INSERT INTO refs(rowid, content) VALUES (?, ?)", [
                (i + 1, " ".join(str(source.get(key, "")) for key in ("title", "publisher", "summary", "keywords")))
                for i, source in enumerate(sources)])
            query = " OR ".join('"' + term + '"' for term in terms)
            rows = [sources[row[0] - 1] for row in db.execute(
                "SELECT rowid FROM refs WHERE refs MATCH ? ORDER BY rank, rowid LIMIT ?",
                (query, a.get("max_items", 5)))]
    return {"topic": a["topic"], "count": len(rows), "items": rows,
            "coverage": "Frozen session reference corpus; no live lookup or date-specific disruption verification."}


def ex_triage_logs(a, m):
    al = m.get("aliases", {})
    svc = canon(a["service"], al)
    logs = m["logs"].get(svc)
    if logs is None:
        return {"__error__": f"unknown service '{a['service']}'"}
    win = logs.get(a["time_window"])
    if win is None:
        return {"service": svc, "time_window": a["time_window"], "groups": [], "note": "no entries"}
    return {"service": svc, "time_window": a["time_window"], **win}


def _mins(s):
    h, mm = s.split(":")
    return int(h) * 60 + int(mm)


def _hm(x):
    return f"{x // 60:02d}:{x % 60:02d}"


def ex_find_meeting_slots(a, m):
    al = m.get("aliases", {})
    day = m["calendars"].get(a["date"])
    if day is None:
        return {"date": a["date"], "free_windows": []}
    lo, hi = {"morning": ("09:00", "12:00"), "afternoon": ("13:00", "18:00"),
              "any": ("09:00", "18:00")}[a.get("part_of_day", "any")]
    people = [canon(p, al) for p in a["attendees"]]
    unknown = [p for p in people if p not in day]
    if unknown:
        return {"__error__": f"no calendar for {', '.join(map(str, unknown))}"}
    busy = sorted((_mins(s), _mins(e)) for p in people for s, e in day[p])
    free, cur = [], _mins(lo)
    for s, e in busy:
        if e <= cur:
            continue
        if s > cur:
            free.append((cur, min(s, _mins(hi))))
        cur = max(cur, e)
        if cur >= _mins(hi):
            break
    if cur < _mins(hi):
        free.append((cur, _mins(hi)))
    dur = a["duration_min"]
    wins = [{"start": _hm(s), "end": _hm(e)} for s, e in free if e - s >= dur]
    return {"date": a["date"], "attendees": people, "duration_min": dur, "free_windows": wins}


def ex_search_rentals(a, m):
    al = m.get("aliases", {})
    out = []
    for r in m["rentals"]:
        if not _eq(r["city"], a["city"], al):
            continue
        if "max_rent" in a and r["rent"] > a["max_rent"]:
            continue
        if "bedrooms" in a and r["bedrooms"] < a["bedrooms"]:
            continue
        if a.get("pets_allowed") and not r["pets"]:
            continue
        if a.get("furnished") and not r["furnished"]:
            continue
        if "max_commute_min" in a and r["commute_min"] > a["max_commute_min"]:
            continue
        out.append(r)
    out.sort(key=lambda r: (r["rent"], r["commute_min"]))
    return {"count": len(out), "listings": out, "currency": m.get("currency", "USD")}


def ex_track_order(a, m):
    o = m["orders"].get(str(a["order_id"]).strip().lstrip("#"))
    if o is None:
        return {"__error__": f"no order with id '{a['order_id']}'"}
    return {"order_id": a["order_id"], **o}


def ex_find_recipes(a, m):
    want = {norm(x) for x in a["ingredients"]}
    out = []
    for r in m["recipes"]:
        have = {norm(x) for x in r["ingredients"]}
        if not want <= have:
            continue
        if a.get("diet") and a["diet"] not in r["diet"]:
            continue
        if "max_time_min" in a and r["time_min"] > a["max_time_min"]:
            continue
        out.append(r)
    out.sort(key=lambda r: r["time_min"])
    return {"count": len(out), "recipes": out}


def ex_search_events(a, m):
    al = m.get("aliases", {})
    out = [e for e in m["events"] if _eq(e["city"], a["city"], al) and e["date"] == a["date"]
           and (not a.get("category") or norm(a["category"]) in norm(e["category"]))
           and e["price"] <= a.get("max_price", 1e12)]
    out.sort(key=lambda e: (e["time"], e["price"]))
    return {"count": len(out), "events": out}


EXECUTORS = {
    "search_rentals": ex_search_rentals, "track_order": ex_track_order,
    "find_recipes": ex_find_recipes, "search_events": ex_search_events,
    "search_flights": ex_search_flights, "search_hotels": ex_search_hotels,
    "compare_products": ex_compare_products, "search_products": ex_search_products,
    "search_restaurants": ex_search_restaurants, "analyze_sales": ex_analyze_sales,
    "compare_quotes": ex_compare_quotes, "summarize_document": ex_summarize_document,
    "generate_report": ex_generate_report, "research_sources": ex_research_sources,
    "triage_logs": ex_triage_logs, "find_meeting_slots": ex_find_meeting_slots,
}
PARTIAL_CAPABLE = {"compare_quotes"}


REQUIRED_MATERIALS = {
    "search_flights": ("flights", list),
    "search_hotels": ("hotels", list),
    "compare_products": ("products", list),
    "search_products": ("catalog", list),
    "search_restaurants": ("restaurants", list),
    "analyze_sales": ("sales_rows", list),
    "compare_quotes": ("vendors", list),
    "summarize_document": ("files", dict),
    "generate_report": ("report", dict),
    "research_sources": ("sources", list),
    "triage_logs": ("logs", dict),
    "find_meeting_slots": ("calendars", dict),
    "search_rentals": ("rentals", list),
    "track_order": ("orders", dict),
    "find_recipes": ("recipes", list),
    "search_events": ("events", list),
}


def validate_materials(tool, materials):
    key, expected_type = REQUIRED_MATERIALS[tool]
    if key not in materials:
        return f"tool '{tool}' is unavailable in this session"
    if not isinstance(materials[key], expected_type):
        raise ValueError(f"Invalid materials for {tool}: {key} must be {expected_type.__name__}")
    return None


def catalog_for_prompt():
    lines = []
    for name, s in TOOLS.items():
        ps = ", ".join(f"{k}{'' if k in s['required'] else '?'}:{t}" for k, t in s["params"].items())
        lines.append(f"- {name}({ps}): {s['description']}")
    return "\n".join(lines)
