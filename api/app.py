from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, List, Optional
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from core.database import init_db, get_session, Repository, TrainingRun, DatasetVersion
from core.logging import get_logger
from core.settings import load_config
logger = get_logger("api")
class IngestRequest(BaseModel):
    url: str
    branch: str = "main"
    keep_clone: bool = False
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1, le=200)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    do_sample: bool = True
    stream: bool = False
class CompleteCodeRequest(BaseModel):
    code_prefix: str
    language: str = "python"
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
class ExplainCodeRequest(BaseModel):
    code: str
    language: str = "python"
class TrainRequest(BaseModel):
    run_name: Optional[str] = None
    dataset_path: Optional[str] = None
    resume_from: Optional[str] = None
class FinetuneRequest(BaseModel):
    model_path: str
    samples: Optional[List[dict]] = None
    dataset_path: Optional[str] = None
    output_dir: Optional[str] = None
    merge_adapters: bool = False
class AddDataRequest(BaseModel):
    raw_dir: str = "data/raw"
    rebuild_dataset: bool = True
    retrain: bool = False
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("CodeLLM API started")
    yield
    logger.info("CodeLLM API shutting down")
app = FastAPI(
    title="CodeLLM Training System",
    description="End-to-end LLM training system for code from GitHub repositories",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_model_path: Optional[str] = None
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}
@app.post("/ingest")
async def ingest_repo(req: IngestRequest, background_tasks: BackgroundTasks):
    def _run():
        from core.ingestion import ingest_repository
        cfg = load_config()
        return ingest_repository(req.url, req.branch, config=cfg, keep_clone=req.keep_clone)
    background_tasks.add_task(_run)
    return {"status": "ingestion started", "url": req.url}
@app.post("/ingest/sync")
async def ingest_repo_sync(req: IngestRequest):
    from core.ingestion import ingest_repository
    cfg = load_config()
    result = ingest_repository(req.url, req.branch, config=cfg, keep_clone=req.keep_clone)
    return result
@app.get("/repositories")
async def list_repositories():
    session = get_session()
    repos = session.query(Repository).order_by(Repository.created_at.desc()).all()
    result = [
        {
            "id": r.id, "url": r.url, "name": r.name, "branch": r.branch,
            "status": r.status, "files_count": r.files_count,
            "total_lines": r.total_lines, "size_kb": r.size_kb,
            "created_at": r.created_at.isoformat(),
        }
        for r in repos
    ]
    session.close()
    return result
@app.post("/dataset/build")
async def build_dataset(background_tasks: BackgroundTasks, incremental: bool = False):
    def _run():
        from pipeline.dataset_builder import build_dataset, get_latest_dataset
        cfg = load_config()
        existing = get_latest_dataset() if incremental else None
        return build_dataset(config=cfg, incremental=incremental, existing_dataset_path=existing)
    background_tasks.add_task(_run)
    return {"status": "dataset build started", "incremental": incremental}
@app.post("/dataset/build/sync")
async def build_dataset_sync(incremental: bool = False):
    from pipeline.dataset_builder import build_dataset, get_latest_dataset
    cfg = load_config()
    existing = get_latest_dataset() if incremental else None
    result = build_dataset(config=cfg, incremental=incremental, existing_dataset_path=existing)
    return result
@app.get("/datasets")
async def list_datasets():
    session = get_session()
    versions = session.query(DatasetVersion).order_by(DatasetVersion.created_at.desc()).all()
    result = [
        {
            "id": v.id, "version": v.version, "path": v.path,
            "total_samples": v.total_samples, "train_samples": v.train_samples,
            "val_samples": v.val_samples, "languages": v.languages,
            "created_at": v.created_at.isoformat(),
        }
        for v in versions
    ]
    session.close()
    return result
@app.post("/train")
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    def _run():
        from training.trainer import train
        cfg = load_config()
        return train(cfg, req.dataset_path, req.run_name, req.resume_from)
    background_tasks.add_task(_run)
    return {"status": "training started", "run_name": req.run_name}
@app.post("/finetune")
async def finetune_model(req: FinetuneRequest, background_tasks: BackgroundTasks):
    def _run():
        from training.finetuner import finetune
        cfg = load_config()
        return finetune(
            req.model_path, req.samples, req.dataset_path,
            req.output_dir, cfg, req.merge_adapters,
        )
    background_tasks.add_task(_run)
    return {"status": "fine-tuning started"}
@app.get("/training/runs")
async def list_training_runs():
    session = get_session()
    runs = session.query(TrainingRun).order_by(TrainingRun.created_at.desc()).all()
    result = [
        {
            "id": r.id, "name": r.name, "status": r.status, "base_model": r.base_model,
            "output_path": r.output_path, "total_samples": r.total_samples,
            "epochs_completed": r.epochs_completed, "metrics": r.metrics,
            "created_at": r.created_at.isoformat(),
        }
        for r in runs
    ]
    session.close()
    return result
@app.post("/model/load")
async def load_model(model_path: str):
    global _model_path
    from inference.engine import get_engine
    try:
        engine = get_engine(model_path)
        _model_path = model_path
        return {"status": "model loaded", "model_path": model_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/generate")
async def generate(req: GenerateRequest):
    from inference.engine import get_engine
    cfg = load_config()
    engine = get_engine(_model_path, cfg)
    if req.stream:
        async def stream_generator():
            for token in engine.stream(req.prompt, req.max_new_tokens, req.temperature, req.top_p):
                yield token
        return StreamingResponse(stream_generator(), media_type="text/plain")
    result = engine.generate(
        req.prompt, req.max_new_tokens, req.temperature,
        req.top_p, req.top_k, req.repetition_penalty, req.do_sample,
    )
    return result
@app.post("/complete")
async def complete_code(req: CompleteCodeRequest):
    from inference.engine import get_engine
    cfg = load_config()
    engine = get_engine(_model_path, cfg)
    return engine.complete_code(req.code_prefix, req.language, req.max_new_tokens)
@app.post("/explain")
async def explain_code(req: ExplainCodeRequest):
    from inference.engine import get_engine
    cfg = load_config()
    engine = get_engine(_model_path, cfg)
    return engine.explain_code(req.code, req.language)
@app.post("/data/add")
async def add_runtime_data(req: AddDataRequest, background_tasks: BackgroundTasks):
    def _run():
        from pipeline.dataset_builder import build_dataset, get_latest_dataset
        cfg = load_config()
        if req.rebuild_dataset:
            existing = get_latest_dataset() if True else None
            build_dataset(config=cfg, incremental=True, existing_dataset_path=existing)
        if req.retrain:
            from training.trainer import train
            train(cfg)
    background_tasks.add_task(_run)
    return {"status": "data addition started", "rebuild_dataset": req.rebuild_dataset, "retrain": req.retrain}
@app.get("/checkpoints")
async def list_checkpoints(model_dir: str = "outputs/model"):
    from training.checkpoint_manager import list_checkpoints as _list
    return _list(model_dir)
@app.post("/export")
async def export_model(model_path: str, export_dir: str, merge_peft: bool = True):
    from training.checkpoint_manager import export_model as _export
    result = _export(model_path, export_dir, merge_peft)
    return {"exported_path": result}
def run_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run("api.app:app", host=host, port=port, reload=False)