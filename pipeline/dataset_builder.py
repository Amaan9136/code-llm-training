from __future__ import annotations
import json
import uuid
from collections import Counter
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple
import datasets as hf_datasets
from core.logging import get_logger
from core.preprocessor import preprocess_file
from core.settings import load_config
logger = get_logger("dataset_builder")
def iter_raw_files(raw_dir: str) -> Generator[Tuple[str, str, str], None, None]:
    raw_path = Path(raw_dir)
    for repo_dir in raw_path.iterdir():
        if not repo_dir.is_dir():
            continue
        for lang_dir in repo_dir.iterdir():
            if not lang_dir.is_dir():
                continue
            language = lang_dir.name
            for file_path in lang_dir.rglob("*"):
                if not file_path.is_file():
                    continue
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    yield language, content, str(file_path)
                except Exception:
                    continue
def build_dataset(
    raw_dir: str = "data/raw",
    output_dir: str = "data/dataset",
    config: Optional[dict] = None,
    version: str = None,
    incremental: bool = False,
    existing_dataset_path: Optional[str] = None,
) -> dict:
    from core.database import get_session, DatasetVersion
    cfg = config or load_config()
    data_cfg = cfg.get("data", {})
    train_split = data_cfg.get("train_split", 0.9)
    val_split = data_cfg.get("val_split", 0.05)
    version = version or str(uuid.uuid4())[:8]
    output_path = Path(output_dir) / version
    output_path.mkdir(parents=True, exist_ok=True)
    all_samples: List[dict] = []
    language_counts: Counter = Counter()
    logger.info(f"Building dataset from {raw_dir}")
    for language, content, file_path in iter_raw_files(raw_dir):
        samples = preprocess_file(content, language, cfg)
        for s in samples:
            s["id"] = str(uuid.uuid4())
            s["source_file"] = file_path
        all_samples.extend(samples)
        language_counts[language] += len(samples)
    if not all_samples:
        raise ValueError(f"No samples found in {raw_dir}")
    if incremental and existing_dataset_path:
        existing = _load_existing_samples(existing_dataset_path)
        existing_ids = {s["text"][:100] for s in existing}
        new_samples = [s for s in all_samples if s["text"][:100] not in existing_ids]
        all_samples = existing + new_samples
        logger.info(f"Incremental: added {len(new_samples)} new samples, total {len(all_samples)}")
    import random
    random.shuffle(all_samples)
    n = len(all_samples)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train: n_train + n_val]
    test_samples = all_samples[n_train + n_val:]
    for split_name, split_data in [("train", train_samples), ("validation", val_samples), ("test", test_samples)]:
        _save_split(split_data, output_path / f"{split_name}.jsonl")
    meta = {
        "version": version,
        "total_samples": n,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "languages": dict(language_counts),
        "output_path": str(output_path),
    }
    (output_path / "metadata.json").write_text(json.dumps(meta, indent=2))
    session = get_session()
    db_version = DatasetVersion(
        version=version,
        path=str(output_path),
        total_samples=n,
        train_samples=len(train_samples),
        val_samples=len(val_samples),
        test_samples=len(test_samples),
        languages=dict(language_counts),
    )
    session.add(db_version)
    session.commit()
    session.close()
    logger.info(f"Dataset v{version}: {n} samples → {output_path}")
    return meta
def _save_split(samples: List[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
def _load_existing_samples(dataset_path: str) -> List[dict]:
    samples = []
    dataset_path = Path(dataset_path)
    for split_file in dataset_path.glob("*.jsonl"):
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return samples
def load_hf_dataset(dataset_path: str) -> hf_datasets.DatasetDict:
    train_file = str(Path(dataset_path) / "train.jsonl")
    val_file = str(Path(dataset_path) / "validation.jsonl")
    test_file = str(Path(dataset_path) / "test.jsonl")
    data_files = {"train": train_file}
    if Path(val_file).exists():
        data_files["validation"] = val_file
    if Path(test_file).exists():
        data_files["test"] = test_file
    return hf_datasets.load_dataset("json", data_files=data_files)
def get_latest_dataset(dataset_dir: str = "data/dataset") -> Optional[str]:
    from core.database import get_session, DatasetVersion
    session = get_session()
    latest = session.query(DatasetVersion).order_by(DatasetVersion.created_at.desc()).first()
    session.close()
    if latest and Path(latest.path).exists():
        return latest.path
    dataset_path = Path(dataset_dir)
    if dataset_path.exists():
        versions = sorted(
            [d for d in dataset_path.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime, reverse=True,
        )
        if versions:
            return str(versions[0])
    return None