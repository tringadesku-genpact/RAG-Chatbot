import re

_ws = re.compile(r"[ \t]+")
_many_newlines = re.compile(r"\n{3,}")

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ws.sub(" ", text)
    text = _many_newlines.sub("\n\n", text)
    return text.strip()
