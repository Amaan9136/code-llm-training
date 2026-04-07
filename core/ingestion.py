from __future__ import annotations
import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple
import git
from core.logging import get_logger
from core.settings import load_config
logger = get_logger("ingestion")
LANGUAGE_MAP: Dict[str, str] = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".jsx": "javascript", ".tsx": "typescript", ".java": "java",
    ".cpp": "cpp", ".c": "c", ".h": "c", ".hpp": "cpp",
    ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
    ".swift": "swift", ".kt": "kotlin", ".scala": "scala",
    ".sh": "shell", ".bash": "shell", ".sql": "sql",
    ".r": "r", ".lua": "lua", ".cs": "csharp", ".fs": "fsharp",
    ".hs": "haskell", ".ex": "elixir", ".exs": "elixir",
    ".clj": "clojure", ".tf": "terraform", ".yaml": "yaml", ".yml": "yaml",
}
def detect_language(file_path: str) -> Optional[str]:
    return LANGUAGE_MAP.get(Path(file_path).suffix.lower())
def is_excluded(path: str, exclude_patterns: List[str]) -> bool:
    path_str = str(path)
    return any(pattern in path_str for pattern in exclude_patterns)
def compute_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
def clone_repository(url: str, branch: str = "main", target_dir: Optional[str] = None) -> str:
    target = target_dir or tempfile.mkdtemp(prefix="codellm_repo_")
    logger.info(f"Cloning {url} (branch: {branch}) → {target}")
    try:
        git.Repo.clone_from(url, target, branch=branch, depth=1, single_branch=True)
    except git.exc.GitCommandError:
        try:
            git.Repo.clone_from(url, target, depth=1)
        except git.exc.GitCommandError as e:
            raise RuntimeError(f"Failed to clone repository: {e}")
    return target
def extract_files(
    repo_path: str,
    config: Optional[dict] = None,
) -> Generator[Tuple[str, str, str, dict], None, None]:
    cfg = config or load_config()
    ingestion_cfg = cfg.get("ingestion", {})
    data_cfg = cfg.get("data", {})
    supported_exts = set(ingestion_cfg.get("supported_extensions", list(LANGUAGE_MAP.keys())))
    exclude_patterns = ingestion_cfg.get("exclude_patterns", [])
    max_size_kb = data_cfg.get("max_file_size_kb", 500)
    min_lines = data_cfg.get("min_file_lines", 5)
    repo_path = Path(repo_path)
    for file_path in repo_path.rglob("*"):
        if not file_path.is_file():
            continue
        if is_excluded(str(file_path.relative_to(repo_path)), exclude_patterns):
            continue
        if file_path.suffix.lower() not in supported_exts:
            continue
        size_kb = file_path.stat().st_size / 1024
        if size_kb > max_size_kb:
            continue
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        lines = content.splitlines()
        if len(lines) < min_lines:
            continue
        language = detect_language(str(file_path))
        if not language:
            continue
        rel_path = str(file_path.relative_to(repo_path))
        content_hash = compute_hash(content)
        meta = {
            "size_kb": round(size_kb, 3),
            "line_count": len(lines),
            "extension": file_path.suffix.lower(),
        }
        yield rel_path, language, content, meta
def ingest_repository(
    url: str,
    branch: str = "main",
    output_dir: str = "data/raw",
    config: Optional[dict] = None,
    keep_clone: bool = False,
) -> dict:
    from core.database import get_session, Repository, CodeFile
    cfg = config or load_config()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    repo_name = re.sub(r"[^\w\-]", "_", url.rstrip("/").split("/")[-1])
    repo_dir = output_path / repo_name
    session = get_session()
    db_repo = session.query(Repository).filter_by(url=url).first()
    if not db_repo:
        db_repo = Repository(url=url, name=repo_name, branch=branch, status="cloning")
        session.add(db_repo)
        session.commit()
    clone_target = tempfile.mkdtemp(prefix="codellm_clone_")
    try:
        clone_repository(url, branch, clone_target)
        db_repo.status = "extracting"
        session.commit()
        repo_dir.mkdir(parents=True, exist_ok=True)
        files_processed = 0
        total_lines = 0
        total_size = 0.0
        seen_hashes = set()
        for rel_path, language, content, meta in extract_files(clone_target, cfg):
            h = meta.get("content_hash") or compute_hash(content)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            lang_dir = repo_dir / language
            lang_dir.mkdir(parents=True, exist_ok=True)
            safe_name = rel_path.replace("/", "_").replace("\\", "_")
            dest = lang_dir / safe_name
            dest.write_text(content, encoding="utf-8")
            existing = session.query(CodeFile).filter_by(
                repository_id=db_repo.id, file_path=rel_path
            ).first()
            if not existing:
                code_file = CodeFile(
                    repository_id=db_repo.id,
                    file_path=rel_path,
                    language=language,
                    content_hash=compute_hash(content),
                    size_kb=meta["size_kb"],
                    line_count=meta["line_count"],
                )
                session.add(code_file)
            files_processed += 1
            total_lines += meta["line_count"]
            total_size += meta["size_kb"]
        db_repo.status = "completed"
        db_repo.files_count = files_processed
        db_repo.total_lines = total_lines
        db_repo.size_kb = total_size
        db_repo.local_path = str(repo_dir)
        session.commit()
        logger.info(f"Ingested {files_processed} files ({total_lines} lines) from {url}")
        return {
            "repository_id": db_repo.id,
            "name": repo_name,
            "url": url,
            "files": files_processed,
            "lines": total_lines,
            "size_kb": total_size,
            "output_path": str(repo_dir),
        }
    except Exception as e:
        db_repo.status = "failed"
        db_repo.error_message = str(e)
        session.commit()
        logger.error(f"Ingestion failed for {url}: {e}")
        raise
    finally:
        session.close()
        if not keep_clone:
            shutil.rmtree(clone_target, ignore_errors=True)