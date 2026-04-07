#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import typer
from rich.console import Console
app = typer.Typer(help="Train and fine-tune LLMs on code data")
console = Console()
@app.command()
def train(
    config: str = typer.Option("config/training.yaml", "--config", "-c"),
    dataset_path: str = typer.Option(None, "--dataset", "-d"),
    run_name: str = typer.Option(None, "--name", "-n"),
    resume_from: str = typer.Option(None, "--resume", "-r"),
):
    from training.trainer import train as _train
    from core.database import init_db
    from core.settings import load_config
    init_db()
    cfg = load_config(config)
    console.print(f"[bold green]Starting training run:[/bold green] {run_name or 'auto'}")
    try:
        result = _train(cfg, dataset_path, run_name, resume_from)
        console.print(f"[bold green]Training complete![/bold green]")
        console.print(f"Model saved to: {result['output_path']}")
        console.print(f"Metrics: {result.get('metrics', {})}")
    except Exception as e:
        console.print(f"[red]Training failed:[/red] {e}")
        raise typer.Exit(1)
@app.command()
def finetune(
    model_path: str = typer.Argument(..., help="Path to base model"),
    dataset_path: str = typer.Option(None, "--dataset", "-d"),
    output_dir: str = typer.Option(None, "--output", "-o"),
    merge: bool = typer.Option(False, "--merge", help="Merge LoRA adapters"),
    config: str = typer.Option("config/training.yaml", "--config", "-c"),
):
    from training.finetuner import finetune as _finetune
    from core.database import init_db
    from core.settings import load_config
    init_db()
    cfg = load_config(config)
    console.print(f"[bold green]Fine-tuning:[/bold green] {model_path}")
    try:
        result = _finetune(model_path, None, dataset_path, output_dir, cfg, merge)
        console.print(f"[bold green]Fine-tuning complete![/bold green]")
        console.print(f"Model saved to: {result['output_path']}")
    except Exception as e:
        console.print(f"[red]Fine-tuning failed:[/red] {e}")
        raise typer.Exit(1)
@app.command()
def build_dataset(
    raw_dir: str = typer.Option("data/raw", "--raw-dir"),
    output_dir: str = typer.Option("data/dataset", "--output"),
    incremental: bool = typer.Option(False, "--incremental", "-i"),
    config: str = typer.Option("config/training.yaml", "--config", "-c"),
):
    from pipeline.dataset_builder import build_dataset as _build, get_latest_dataset
    from core.database import init_db
    from core.settings import load_config
    init_db()
    cfg = load_config(config)
    console.print(f"[bold cyan]Building dataset from[/bold cyan] {raw_dir}")
    existing = get_latest_dataset() if incremental else None
    try:
        result = _build(raw_dir, output_dir, cfg, incremental=incremental, existing_dataset_path=existing)
        console.print(f"[bold green]Dataset built![/bold green]")
        console.print(f"Version: {result['version']}")
        console.print(f"Total samples: {result['total_samples']}")
        console.print(f"Languages: {result['languages']}")
    except Exception as e:
        console.print(f"[red]Dataset build failed:[/red] {e}")
        raise typer.Exit(1)
@app.command()
def list_runs():
    from core.database import init_db, get_session, TrainingRun
    from rich.table import Table
    init_db()
    session = get_session()
    runs = session.query(TrainingRun).order_by(TrainingRun.created_at.desc()).all()
    session.close()
    if not runs:
        console.print("[yellow]No training runs found[/yellow]")
        return
    table = Table(title="Training Runs")
    for col in ["Name", "Status", "Model", "Samples", "Output"]:
        table.add_column(col)
    for r in runs:
        table.add_row(r.name, r.status, r.base_model[:40], str(r.total_samples), r.output_path or "-")
    console.print(table)
if __name__ == "__main__":
    app()