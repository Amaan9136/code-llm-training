#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import typer
from rich.console import Console
from rich.table import Table
app = typer.Typer(help="Ingest GitHub repositories for training data")
console = Console()
@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    repo_url: str = typer.Option(None, "--repo", help="GitHub repository URL"),
    branch: str = typer.Option("main", "--branch", "-b", help="Branch to clone"),
    output_dir: str = typer.Option("data/raw", "--output", "-o", help="Output directory"),
    keep_clone: bool = typer.Option(False, "--keep-clone", help="Keep cloned repository"),
):
    if ctx.invoked_subcommand is not None:
        return
    if not repo_url:
        console.print(ctx.get_help())
        raise typer.Exit()
    from core.ingestion import ingest_repository
    from core.database import init_db
    from core.settings import load_config
    init_db()
    cfg = load_config()
    console.print(f"[bold cyan]Ingesting[/bold cyan] {repo_url} (branch: {branch})")
    try:
        result = ingest_repository(repo_url, branch, output_dir, cfg, keep_clone)
        table = Table(title="Ingestion Result")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")
        for k, v in result.items():
            table.add_row(str(k), str(v))
        console.print(table)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
@app.command()
def list_repos():
    from core.database import init_db, get_session, Repository
    init_db()
    session = get_session()
    repos = session.query(Repository).all()
    session.close()
    if not repos:
        console.print("[yellow]No repositories ingested yet[/yellow]")
        return
    table = Table(title="Ingested Repositories")
    for col in ["Name", "URL", "Status", "Files", "Lines"]:
        table.add_column(col)
    for r in repos:
        table.add_row(r.name, r.url[:60], r.status, str(r.files_count), str(r.total_lines))
    console.print(table)
if __name__ == "__main__":
    app()