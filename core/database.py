from __future__ import annotations
import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import create_engine, String, DateTime, Integer, Float, Text, Boolean, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from sqlalchemy.sql import func
from core.settings import settings
class Base(DeclarativeBase):
    pass
class Repository(Base):
    __tablename__ = "repositories"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    url: Mapped[str] = mapped_column(String(512), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    branch: Mapped[str] = mapped_column(String(128), default="main")
    status: Mapped[str] = mapped_column(String(64), default="pending")
    files_count: Mapped[int] = mapped_column(Integer, default=0)
    total_lines: Mapped[int] = mapped_column(Integer, default=0)
    size_kb: Mapped[float] = mapped_column(Float, default=0.0)
    local_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
class CodeFile(Base):
    __tablename__ = "code_files"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    repository_id: Mapped[str] = mapped_column(String(36), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    language: Mapped[str] = mapped_column(String(64), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    size_kb: Mapped[float] = mapped_column(Float, default=0.0)
    line_count: Mapped[int] = mapped_column(Integer, default=0)
    is_processed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
class TrainingRun(Base):
    __tablename__ = "training_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(64), default="pending")
    base_model: Mapped[str] = mapped_column(String(256), nullable=False)
    output_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    total_samples: Mapped[int] = mapped_column(Integer, default=0)
    epochs_completed: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    path: Mapped[str] = mapped_column(String(512), nullable=False)
    total_samples: Mapped[int] = mapped_column(Integer, default=0)
    train_samples: Mapped[int] = mapped_column(Integer, default=0)
    val_samples: Mapped[int] = mapped_column(Integer, default=0)
    test_samples: Mapped[int] = mapped_column(Integer, default=0)
    languages: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
def init_db():
    Base.metadata.create_all(bind=engine)
def get_session() -> Session:
    return Session(engine)