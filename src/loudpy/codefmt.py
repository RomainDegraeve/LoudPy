"""
Readable generated code: a line longer than the width has the arguments of its call (or the items of its list) one
per line, nested calls wrapped the same way; lists of numbers fill their lines. Strings are never cut.

    text = "\\n".join(wrap_lines(lines, width=100))
"""
from __future__ import annotations

OPEN, CLOSE = "([{", ")]}"


def _close_of(s: str, i: int) -> int:
    """The index of the bracket closing s[i] (-1: none), strings skipped."""
    depth, quote, k = 0, None, i
    while k < len(s):
        ch = s[k]
        if quote:
            if ch == "\\":
                k += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch in OPEN:
            depth += 1
        elif ch in CLOSE:
            depth -= 1
            if depth == 0:
                return k
        k += 1
    return -1


def _split_top(s: str) -> list[str]:
    """s split at its top-level commas (not inside brackets or strings)."""
    parts, depth, quote, cur, k = [], 0, None, "", 0
    while k < len(s):
        ch = s[k]
        if quote:
            cur += ch
            if ch == "\\" and k + 1 < len(s):
                cur += s[k + 1]
                k += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            cur += ch
        elif ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            depth += ch in OPEN
            depth -= ch in CLOSE
            cur += ch
        k += 1
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _group(body: str) -> tuple[int, int] | None:
    """The first bracket group (outside strings) that ends the line (but for a trailing comma)."""
    quote, k = None, 0
    while k < len(body):
        ch = body[k]
        if quote:
            if ch == "\\":
                k += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch in OPEN:
            j = _close_of(body, k)
            if j < 0:
                return None
            if body[j + 1:].strip() in ("", ","):
                return k, j
            k = j
        k += 1
    return None


def wrap(line: str, width: int = 100) -> list[str]:
    """One line of code as lines of at most `width` characters where it can be cut (see the module)."""
    if len(line) <= width:
        return [line]
    indent = line[:len(line) - len(line.lstrip())]
    body = line.strip()
    found = _group(body)
    if found is None:
        return [line]
    k, j = found
    head, inner, tail = body[:k + 1], body[k + 1:j], body[j:]
    parts = _split_top(inner)
    if not parts:
        return [line]
    inner_indent = indent + "    "
    if body[k] == "[" and not any(c in p for p in parts for c in OPEN):     # numbers, names: fill the lines
        rows, cur = [], ""
        for p in parts:
            if cur and len(inner_indent) + len(cur) + len(p) + 1 > width:
                rows.append(inner_indent + cur.rstrip())
                cur = ""
            cur += p + ", "
        rows.append(inner_indent + cur.rstrip())
        return [indent + head, *rows, indent + tail]
    out = [indent + head]
    for p in parts:
        out += wrap(inner_indent + p + ",", width)
    return out + [indent + tail]


def wrap_lines(lines, width: int = 100) -> list[str]:
    """Every line wrapped (a line holding several lines is split first)."""
    out: list[str] = []
    for text in lines:
        for line in str(text).split("\n"):
            out += wrap(line, width)
    return out
