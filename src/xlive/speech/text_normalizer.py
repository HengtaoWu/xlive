"""Prepare display-oriented model text for natural speech synthesis."""

from __future__ import annotations

import re


_LATEX_WRAPPERS = re.compile(
    r"\$\$(.+?)\$\$|\$(.+?)\$|\\\[(.+?)\\\]|\\\((.+?)\\\)",
    re.DOTALL,
)
_FRAC = re.compile(r"\\(?:d|t)?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
_SQRT = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
_POWER = re.compile(r"\^\s*\{([^{}]+)\}|\^\s*([\w\d]+)")
_SUBSCRIPT = re.compile(r"_\s*\{([^{}]+)\}|_\s*([\w\d]+)")
_INLINE_DIVISION = re.compile(r"(?<=[A-Za-z0-9\)])\s*/\s*(?=[A-Za-z0-9\(])")
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)]\([^\s)]+\)")
_CODE_FENCE = re.compile(r"```(?:\w+)?\s*|```")
_URL = re.compile(r"https?://[^\s，。！？；]+")
_MATH_FRAGMENT = re.compile("\ufff0(.+?)\ufff1", re.DOTALL)

_COMMANDS = {
    r"\times": "乘以",
    r"\cdot": "乘以",
    r"\div": "除以",
    r"\pm": "正负",
    r"\leq": "小于等于",
    r"\le": "小于等于",
    r"\geq": "大于等于",
    r"\ge": "大于等于",
    r"\neq": "不等于",
    r"\approx": "约等于",
    r"\to": "趋近于",
    r"\infty": "无穷大",
    r"\pi": "派",
    r"\theta": "西塔",
    r"\alpha": "阿尔法",
    r"\beta": "贝塔",
    r"\gamma": "伽马",
    r"\Delta": "德尔塔",
    r"\delta": "德尔塔",
}

_SYMBOLS = {
    "≤": "小于等于",
    "≥": "大于等于",
    "≠": "不等于",
    "≈": "约等于",
    "→": "趋近于",
    "∞": "无穷大",
    "±": "正负",
    "×": "乘以",
    "÷": "除以",
    "·": "乘以",
    "=": "等于",
}

_POWER_READINGS = (
    ("⁰", "的零次方", "to the zeroth power"),
    ("¹", "的一次方", "to the first power"),
    ("²", "的平方", "squared"),
    ("³", "的立方", "cubed"),
    ("⁴", "的四次方", "to the fourth power"),
    ("⁵", "的五次方", "to the fifth power"),
    ("⁶", "的六次方", "to the sixth power"),
    ("⁷", "的七次方", "to the seventh power"),
    ("⁸", "的八次方", "to the eighth power"),
    ("⁹", "的九次方", "to the ninth power"),
)
_SUPERSCRIPTS = str.maketrans({symbol: zh for symbol, zh, _ in _POWER_READINGS})
_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "零一二三四五六七八九")

_ENGLISH_SPEECH = {
    "小于等于": " less than or equal to ",
    "大于等于": " greater than or equal to ",
    "不等于": " not equal to ",
    "约等于": " approximately equals ",
    "趋近于": " approaches ",
    "无穷大": " infinity ",
    **{zh: f" {en} " for _, zh, en in _POWER_READINGS},
    "正负": " plus or minus ",
    "乘以": " times ",
    "除以": " divided by ",
    "等于": " equals ",
    "根号": " square root of ",
    "下标": " subscript ",
    "次方": " power ",
    "阿尔法": " alpha ",
    "贝塔": " beta ",
    "伽马": " gamma ",
    "德尔塔": " delta ",
    "西塔": " theta ",
    "派": " pi ",
    "加": " plus ",
    "减": " minus ",
}


def normalize_text_for_speech(text: str) -> str:
    """Return a spoken copy of text without changing its displayed form.

    The conversion deliberately targets common educational notation instead of
    attempting to evaluate arbitrary LaTeX. Unknown commands are reduced to
    their readable names, so the TTS engine never receives raw control syntax.
    """

    if not text:
        return text
    speak_chinese = re.search(r"[\u3400-\u9fff]", text) is not None

    urls: list[str] = []

    def preserve_url(match: re.Match[str]) -> str:
        urls.append(match.group(0))
        return f"URLPLACEHOLDER{len(urls) - 1}TOKEN"

    value = _URL.sub(preserve_url, text)
    value = _CODE_FENCE.sub("", value)
    value = _MARKDOWN_LINK.sub(r"\1", value)
    value = _normalize_latex(value)

    value = value.translate(_SUPERSCRIPTS).translate(_SUBSCRIPTS)
    value = _INLINE_DIVISION.sub("除以", value)
    for symbol, spoken in _SYMBOLS.items():
        value = value.replace(symbol, spoken)

    value = re.sub(r"(?<=[A-Za-z0-9)])\s*\*\s*(?=[A-Za-z0-9(])", "乘以", value)
    # A lone plus is mathematical; repeated pluses are usually source code
    # (for example C++) and should remain untouched.
    value = re.sub(r"(?<!\+)\s*\+\s*(?!\+)", "加", value)
    value = value.replace("−", "减")
    value = re.sub(r"(?<=\s)-(?=\s)|(?<=\d)-(?=\d)", "减", value)
    value = value.replace("{", "").replace("}", "")
    value = re.sub(r"[`*_#]", "", value)
    value = _MATH_FRAGMENT.sub(lambda match: _speak_implicit_products(match.group(1)), value)
    if not speak_chinese:
        for chinese, english in _ENGLISH_SPEECH.items():
            value = value.replace(chinese, english)
    value = re.sub(r"\s+", " ", value)
    for index, url in enumerate(urls):
        value = value.replace(f"URLPLACEHOLDER{index}TOKEN", url)
    return value.strip()


def _normalize_latex(value: str) -> str:
    value = _LATEX_WRAPPERS.sub(_unwrap_math, value)

    # Fractions can be nested one level in normal classroom formulas. Repeating
    # the substitution handles the common inner-first case without a parser.
    for _ in range(3):
        replaced = _FRAC.sub(lambda match: f"{match.group(1)}除以{match.group(2)}", value)
        if replaced == value:
            break
        value = replaced
    value = _SQRT.sub(lambda match: f"根号{match.group(1)}", value)
    value = _POWER.sub(_speak_power, value)
    value = _SUBSCRIPT.sub(lambda match: f"下标{match.group(1) or match.group(2)}", value)

    for command, spoken in _COMMANDS.items():
        value = value.replace(command, spoken)
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = re.sub(r"\\(?:mathrm|mathbf|text)\s*\{([^{}]+)\}", r"\1", value)
    value = re.sub(r"\\([A-Za-z]+)", r"\1", value)
    return value


def _unwrap_math(match: re.Match[str]) -> str:
    content = next(group for group in match.groups() if group is not None)
    return f"\ufff0{content}\ufff1"


def _speak_power(match: re.Match[str]) -> str:
    exponent = match.group(1) or match.group(2) or ""
    if exponent == "2":
        return "的平方"
    if exponent == "3":
        return "的立方"
    return f"的{exponent}次方"


def _speak_implicit_products(expression: str) -> str:
    """Expand compact school-formula products without touching ordinary prose."""

    value = re.sub(r"(?<=\d)(?=[A-Za-z])", "乘以", expression)
    value = re.sub(r"(?<=\))(?=[A-Za-z0-9])", "乘以", value)

    preserved = {"kg", "cm", "mm", "km", "ms", "Hz", "Pa"}

    def split_pair(match: re.Match[str]) -> str:
        pair = match.group(0)
        return pair if pair in preserved else f"{pair[0]}乘以{pair[1]}"

    return re.sub(r"(?<![A-Za-z])[A-Za-z]{2}(?![A-Za-z])", split_pair, value)
