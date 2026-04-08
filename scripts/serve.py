#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import typer
from rich.console import Console
app = typer.Typer(help="Serve the trained model for inference")
console = Console()
@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    model_path: str = typer.Option(None, "--model-path", "-m"),
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port", "-p"),
    config: str = typer.Option("config/training.yaml", "--config", "-c"),
):
    if ctx.invoked_subcommand is not None:
        return
    if not model_path:
        console.print(ctx.get_help())
        raise typer.Exit()
    from core.database import init_db
    import os
    os.environ["CONFIG_PATH"] = config
    os.environ["MODEL_PATH"] = model_path
    init_db()
    console.print(f"[bold green]Starting CodeLLM API[/bold green]")
    console.print(f"  Model: {model_path}")
    console.print(f"  URL: http://{host}:{port}")
    console.print(f"  Docs: http://{host}:{port}/docs")
    import uvicorn
    uvicorn.run("api.app:app", host=host, port=port, reload=False)
@app.command()
def chat(
    model_path: str = typer.Option("outputs/model", "--model-path", "-m"),
    config: str = typer.Option("config/training.yaml", "--config", "-c"),
):
    from inference.engine import CodeInferenceEngine
    from core.settings import load_config
    cfg = load_config(config)
    console.print(f"[bold cyan]Loading model from[/bold cyan] {model_path}")
    engine = CodeInferenceEngine(model_path, cfg)
    engine.load()
    console.print("[bold green]Model loaded! Type your prompt (Ctrl+C to exit)[/bold green]")
    while True:
        try:
            prompt = console.input("[bold blue]>>> [/bold blue]")
            if prompt.strip().lower() in ("exit", "quit", "q"):
                break
            result = engine.generate(prompt)
            console.print(f"\n[bold green]Generated:[/bold green]")
            console.print(result["generated_text"])
            console.print(f"\n[dim]({result['tokens_generated']} tokens in {result['time_seconds']}s)[/dim]\n")
        except KeyboardInterrupt:
            break
    console.print("[yellow]Goodbye![/yellow]")
@app.command()
def complete(
    model_path: str = typer.Option("outputs/model", "--model-path", "-m"),
    language: str = typer.Option("python", "--language", "-l"),
    prefix: str = typer.Argument(..., help="Code prefix to complete"),
    config: str = typer.Option("config/training.yaml", "--config", "-c"),
):
    from inference.engine import CodeInferenceEngine
    from core.settings import load_config
    cfg = load_config(config)
    engine = CodeInferenceEngine(model_path, cfg)
    engine.load()
    result = engine.complete_code(prefix, language)
    console.print(f"[bold green]Completion:[/bold green]")
    console.print(result["generated_text"])
if __name__ == "__main__":
    app()