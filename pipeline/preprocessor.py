from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple
from core.logging import get_logger
logger = get_logger("preprocessor")
COMMENT_PATTERNS: Dict[str, List[str]] = {
    "python": [r"#.*$", r'"""[\s\S]*?"""', r"'''[\s\S]*?'''"],
    "javascript": [r"//.*$", r"/\*[\s\S]*?\*/"],
    "typescript": [r"//.*$", r"/\*[\s\S]*?\*/"],
    "java": [r"//.*$", r"/\*[\s\S]*?\*/"],
    "cpp": [r"//.*$", r"/\*[\s\S]*?\*/"],
    "c": [r"//.*$", r"/\*[\s\S]*?\*/"],
    "go": [r"//.*$", r"/\*[\s\S]*?\*/"],
    "rust": [r"//.*$", r"/\*[\s\S]*?\*/"],
    "ruby": [r"#.*$"],
    "shell": [r"#.*$"],
    "sql": [r"--.*$", r"/\*[\s\S]*?\*/"],
}
PROMPT_TEMPLATES: Dict[str, str] = {
    "completion": "{code}",
    "instruction": "### Language: {language}\n### Task: Complete the following code\n### Code:\n{code}\n### End",
    "fim": "<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>",
    "docstring": "### Code:\n{code}\n### Docstring:\n{docstring}",
    "explain": "### Code:\n{code}\n### Explanation:\nThis code",
}
def remove_comments(code: str, language: str, preserve_docstrings: bool = True) -> str:
    patterns = COMMENT_PATTERNS.get(language, [])
    if not patterns:
        return code
    result = code
    for i, pattern in enumerate(patterns):
        if preserve_docstrings and language == "python" and i > 0:
            continue
        result = re.sub(pattern, "", result, flags=re.MULTILINE)
    return result
def normalize_whitespace(code: str) -> str:
    lines = code.split("\n")
    normalized = []
    blank_count = 0
    for line in lines:
        stripped = line.rstrip()
        if stripped == "":
            blank_count += 1
            if blank_count <= 2:
                normalized.append("")
        else:
            blank_count = 0
            normalized.append(stripped)
    return "\n".join(normalized).strip()
def remove_boilerplate(code: str, language: str) -> str:
    if language in ("javascript", "typescript"):
        code = re.sub(r'"use strict";?\n?', "", code)
        code = re.sub(r"'use strict';?\n?", "", code)
    if language == "python":
        code = re.sub(r"if __name__ == ['\"]__main__['\"]:?\n(\s+.*\n?)*", "", code)
    return code
def is_meaningful_code(code: str, min_tokens: int = 20) -> bool:
    tokens = re.findall(r"\w+", code)
    return len(tokens) >= min_tokens
def split_into_chunks(
    code: str,
    chunk_size: int = 1024,
    overlap: int = 128,
    unit: str = "chars",
) -> List[str]:
    if unit == "lines":
        lines = code.split("\n")
        chunks = []
        i = 0
        while i < len(lines):
            chunk = "\n".join(lines[i: i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
            i += chunk_size - overlap
        return chunks
    chunks = []
    i = 0
    while i < len(code):
        chunk = code[i: i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks
def extract_functions(code: str, language: str) -> List[Tuple[str, str]]:
    functions = []
    if language == "python":
        pattern = r"(def\s+\w+[^:]*:(?:\n(?:[ \t]+.+|\s*$))*)"
        matches = re.finditer(pattern, code, re.MULTILINE)
        for m in matches:
            func_code = m.group(1).strip()
            name_match = re.match(r"def\s+(\w+)", func_code)
            name = name_match.group(1) if name_match else "unknown"
            functions.append((name, func_code))
    elif language in ("javascript", "typescript"):
        patterns = [
            r"((?:async\s+)?function\s+\w+[^{]*\{(?:[^{}]|\{[^{}]*\})*\})",
            r"(const\s+\w+\s*=\s*(?:async\s+)?(?:\([^)]*\)|\w+)\s*=>[^;]+;)",
        ]
        for pattern in patterns:
            for m in re.finditer(pattern, code, re.MULTILINE | re.DOTALL):
                func_code = m.group(1).strip()
                functions.append(("anonymous", func_code))
    return functions
def format_training_sample(
    code: str,
    language: str,
    template: str = "instruction",
    file_path: str = "",
    **kwargs,
) -> str:
    tmpl = PROMPT_TEMPLATES.get(template, PROMPT_TEMPLATES["instruction"])
    context = {"code": code, "language": language, "file_path": file_path, **kwargs}
    try:
        return tmpl.format(**context)
    except KeyError:
        return code
def preprocess_file(
    content: str,
    language: str,
    config: Optional[dict] = None,
    template: str = "instruction",
) -> List[dict]:
    cfg = config or {}
    data_cfg = cfg.get("data", {})
    chunk_size = data_cfg.get("chunk_size", 1024)
    chunk_overlap = data_cfg.get("chunk_overlap", 128)
    cleaned = normalize_whitespace(content)
    cleaned = remove_boilerplate(cleaned, language)
    if not is_meaningful_code(cleaned):
        return []
    chunks = split_into_chunks(cleaned, chunk_size=chunk_size, overlap=chunk_overlap)
    samples = []
    for chunk in chunks:
        if not is_meaningful_code(chunk, min_tokens=10):
            continue
        text = format_training_sample(chunk, language, template=template)
        samples.append({
            "text": text,
            "language": language,
            "chunk_size": len(chunk),
            "token_estimate": len(chunk.split()) * 1.3,
        })
    return samples