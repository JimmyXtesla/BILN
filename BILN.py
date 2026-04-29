#!/usr/bin/env python3
"""
BILN - Bioinformatician's Interactive Lab Notebook
Author: Jimmy X Banda
Version: 2.0 (2025)

A provenance-tracking CLI for bioinformatics workflows.
Tracks commands, files, lineage, environments, and resources.
"""

import hashlib
import json
import os
import platform
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import psutil
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import track
from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.table import Table

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BILN_DIR        = Path(".biln")
DB_PATH         = BILN_DIR / "biln.db"
ARCHIVE_DIR     = BILN_DIR / "archive"
HASH_CHUNK_SIZE = 8 * 1024 * 1024   # 8 MB chunks for streaming hash
LARGE_FILE_MB   = 100               # threshold for archive candidates
HISTORY_LIMIT   = 15                # default rows shown in history
MAX_LINEAGE_VIZ = 50                # max edges in mermaid / dot graphs
DB_SCHEMA_VER   = 2                 # bump when schema changes

# Optional heavy deps
try:
    import pysam
except ImportError:
    pysam = None

try:
    from jinja2 import Template
    HAS_JINJA = True
except ImportError:
    HAS_JINJA = False

# ---------------------------------------------------------------------------
# App & Console
# ---------------------------------------------------------------------------

app = typer.Typer(
    help=(
        "BILN v2.0 — The Bioinformatician's Interactive Lab Notebook.\n\n"
        "Track commands, files, lineage, and environments reproducibly."
    ),
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name   TEXT    NOT NULL UNIQUE,
    active INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS logs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id   INTEGER NOT NULL REFERENCES projects(id),
    timestamp    TEXT    NOT NULL,
    category     TEXT    NOT NULL,
    content      TEXT,
    cmd          TEXT,
    tool_version TEXT,
    git_hash     TEXT,
    runtime_s    REAL,
    env_info     TEXT,
    exit_code    INTEGER
);

CREATE TABLE IF NOT EXISTS files (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    path       TEXT    NOT NULL,
    hash_md5   TEXT,
    metrics    TEXT,
    archived   INTEGER NOT NULL DEFAULT 0,
    UNIQUE (project_id, path)
);

CREATE TABLE IF NOT EXISTS lineage (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    log_id         INTEGER NOT NULL REFERENCES logs(id),
    input_file_id  INTEGER REFERENCES files(id),
    output_file_id INTEGER REFERENCES files(id)
);

CREATE TABLE IF NOT EXISTS samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  INTEGER NOT NULL REFERENCES projects(id),
    sample_name TEXT,
    condition   TEXT,
    replicate   TEXT,
    file_path   TEXT
);
"""


@contextmanager
def get_db():
    """
    Context manager that yields an open, configured sqlite3 connection
    and commits on clean exit, rolls back on error, then closes.
    """
    BILN_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)

    # Seed schema version if this is a fresh DB
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
        (str(DB_SCHEMA_VER),),
    )
    conn.commit()

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def require_project(conn: sqlite3.Connection) -> Tuple[int, str]:
    """
    Return (project_id, project_name) for the active project.
    Creates a 'default' project automatically if none exists.
    Raises SystemExit with a helpful message if DB is corrupt.
    """
    row = conn.execute(
        "SELECT id, name FROM projects WHERE active = 1"
    ).fetchone()

    if row is None:
        # Auto-create and activate 'default'
        conn.execute(
            "INSERT OR IGNORE INTO projects (name, active) VALUES ('default', 1)"
        )
        conn.commit()
        row = conn.execute(
            "SELECT id, name FROM projects WHERE active = 1"
        ).fetchone()

    if row is None:
        console.print("[red]Error:[/red] No active project and could not create one.")
        raise typer.Exit(code=1)

    return int(row["id"]), str(row["name"])


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------

def file_hash_md5(path: str) -> str:
    """
    Compute MD5 over the *entire* file in streaming chunks.
    Returns 'missing' or 'error:<msg>' on failure.
    """
    p = Path(path)
    if not p.exists():
        return "missing"
    h = hashlib.md5()
    try:
        with open(p, "rb") as fh:
            while chunk := fh.read(HASH_CHUNK_SIZE):
                h.update(chunk)
        return h.hexdigest()
    except OSError as exc:
        return f"error:{exc}"


def bio_file_metrics(path: str) -> str:
    """
    Collect lightweight bioinformatics metrics for known file types.
    Always returns a JSON string (even on failure).
    """
    p = Path(path)
    stats: dict = {}

    if not p.exists():
        return json.dumps({"error": "file_missing"})

    try:
        stats["size_mb"] = round(p.stat().st_size / (1024 ** 2), 3)
    except OSError:
        return json.dumps({"error": "stat_failed"})

    suffix = p.suffix.lower()

    if pysam and suffix in {".bam", ".sam", ".cram"}:
        try:
            with pysam.AlignmentFile(path, "rb") as aln:
                stats["mapped_reads"]   = aln.mapped
                stats["unmapped_reads"] = aln.unmapped
        except Exception:
            stats["bam_error"] = "could_not_read"

    elif suffix in {".fastq", ".fq", ".fastq.gz", ".fq.gz"}:
        # Estimate read count cheaply via line count / 4
        try:
            opener = (
                __import__("gzip").open if suffix.endswith(".gz")
                else open
            )
            with opener(path, "rt") as fh:
                n = sum(1 for _ in fh)
            stats["est_reads"] = n // 4
        except Exception:
            stats["fastq_error"] = "could_not_count"

    elif suffix in {".vcf", ".bcf"}:
        try:
            result = subprocess.run(
                ["grep", "-vc", "^#", path],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                stats["variant_count"] = int(result.stdout.strip())
        except Exception:
            pass

    return json.dumps(stats)


def upsert_file(conn: sqlite3.Connection, project_id: int, path: str) -> int:
    """
    Track a file (insert or update). Returns the file row ID.
    Stores the absolute, resolved path so records survive directory changes.
    """
    abs_path = str(Path(path).resolve())
    h        = file_hash_md5(abs_path)
    metrics  = bio_file_metrics(abs_path)

    conn.execute(
        """
        INSERT INTO files (project_id, path, hash_md5, metrics)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(project_id, path) DO UPDATE SET
            hash_md5 = excluded.hash_md5,
            metrics  = excluded.metrics
        """,
        (project_id, abs_path, h, metrics),
    )
    row = conn.execute(
        "SELECT id FROM files WHERE project_id = ? AND path = ?",
        (project_id, abs_path),
    ).fetchone()
    return int(row["id"])


# ---------------------------------------------------------------------------
# Environment / git helpers
# ---------------------------------------------------------------------------

def git_info() -> str:
    """Return short SHA (+ DIRTY flag) or 'no-git'."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "diff", "--shortstat"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return f"{sha}+dirty" if dirty else sha
    except (subprocess.SubprocessError, FileNotFoundError):
        return "no-git"


def tool_version(cmd_string: str) -> str:
    """
    Try common version flags on the first token of a command string.
    Returns the first line of output, or 'unknown'.
    """
    if not cmd_string:
        return "unknown"
    binary = shlex.split(cmd_string)[0]
    for flag in ("--version", "-version", "-v", "-V"):
        try:
            out = subprocess.check_output(
                [binary, flag],
                stderr=subprocess.STDOUT,
                timeout=2,
            ).decode().strip().splitlines()
            if out:
                return out[0]
        except Exception:
            continue
    return "unknown"


def env_snapshot() -> dict:
    """Capture lightweight environment metadata."""
    return {
        "host":         platform.node(),
        "os":           platform.system(),
        "python":       platform.python_version(),
        "conda_env":    os.environ.get("CONDA_DEFAULT_ENV", ""),
        "slurm_job":    os.environ.get("SLURM_JOB_ID", ""),
        "container":    (
            "docker"      if Path("/.dockerenv").exists()
            else "singularity" if "SINGULARITY_NAME" in os.environ
            else "host"
        ),
    }


# ---------------------------------------------------------------------------
# Snakemake helper
# ---------------------------------------------------------------------------

def _abs_to_rel(abs_path: str, cwd: Path) -> str:
    """Convert absolute path to relative if possible, else keep absolute."""
    try:
        return str(Path(abs_path).relative_to(cwd))
    except ValueError:
        return abs_path


def _safe_shell_cmd(cmd: str) -> str:
    """
    Escape a shell command for embedding inside a Snakemake triple-quoted
    shell directive.  Escapes backslashes and triple-single-quotes only.
    """
    return cmd.replace("\\", "\\\\").replace("'''", r"\'\'\'")


# ---------------------------------------------------------------------------
# PROJECT MANAGEMENT
# ---------------------------------------------------------------------------

@app.command()
def init():
    """Initialise a new BILN environment in the current directory."""
    with get_db() as conn:
        require_project(conn)   # ensures default project exists
    console.print(
        Panel(
            f"[green]BILN v2.0 initialised.[/green]\n"
            f"Database: [dim]{DB_PATH}[/dim]",
            title="BILN Ready",
        )
    )


@app.command()
def project(
    name: str,
    create: bool = typer.Option(False, "--create", "-c", help="Create project if it does not exist"),
):
    """Switch the active project.  Use --create to make a new one."""
    with get_db() as conn:
        exists = conn.execute(
            "SELECT id FROM projects WHERE name = ?", (name,)
        ).fetchone()

        if not exists and not create:
            console.print(
                f"[red]Project '{name}' not found.[/red] "
                "Use [bold]--create[/bold] to make it."
            )
            raise typer.Exit(code=1)

        if not exists:
            conn.execute(
                "INSERT INTO projects (name, active) VALUES (?, 0)", (name,)
            )

        conn.execute("UPDATE projects SET active = 0")
        conn.execute("UPDATE projects SET active = 1 WHERE name = ?", (name,))

    console.print(f"Active project → [bold cyan]{name}[/bold cyan]")


@app.command("list-projects")
def list_projects():
    """List all projects."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT name, active FROM projects ORDER BY name"
        ).fetchall()

    table = Table(title="Projects")
    table.add_column("Name")
    table.add_column("Active")
    for r in rows:
        marker = "[green]✔[/green]" if r["active"] else ""
        table.add_row(r["name"], marker)
    console.print(table)


# ---------------------------------------------------------------------------
# EXECUTION & LOGGING
# ---------------------------------------------------------------------------

@app.command()
def log(
    message: str,
    category: str = typer.Option("note", help="Category label for the entry"),
):
    """Manually record a note or observation."""
    with get_db() as conn:
        p_id, _ = require_project(conn)
        conn.execute(
            "INSERT INTO logs (project_id, timestamp, category, content) VALUES (?,?,?,?)",
            (p_id, datetime.now().isoformat(), category.upper(), message),
        )
    console.print("[dim]Logged.[/dim]")


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(
    ctx: typer.Context,
    inputs:  List[str] = typer.Option([], "--input",  "-i", help="Input  files to track"),
    outputs: List[str] = typer.Option([], "--output", "-o", help="Output files to track"),
):
    """
    Run a shell command while recording git state, tool version,
    environment, runtime, and input/output lineage.

    Example:
        biln run -i ref.fa -i reads.fq -o aln.bam "bwa mem ref.fa reads.fq | samtools sort -o aln.bam"
    """
    cmd = " ".join(ctx.args)
    if not cmd:
        console.print("[red]Error:[/red] No command provided after options.")
        raise typer.Exit(code=1)

    with get_db() as conn:
        p_id, p_name = require_project(conn)

        # --- Track inputs BEFORE execution so we have IDs even if run fails ---
        in_ids: List[int] = []
        for f in inputs:
            if not Path(f).exists():
                console.print(f"[yellow]Warning:[/yellow] Input '{f}' does not exist yet.")
            in_ids.append(upsert_file(conn, p_id, f))

        sha      = git_info()
        ver      = tool_version(cmd)
        env      = env_snapshot()

        console.print(
            Panel(
                f"[bold]Project:[/bold] {p_name}\n"
                f"[bold]CMD:[/bold]     {cmd}\n"
                f"[bold]Env:[/bold]     {env['conda_env'] or 'base'} "
                f"({env['container']})\n"
                f"[bold]Git:[/bold]     {sha}",
                title="BILN Runner",
            )
        )

        t0 = time.perf_counter()
        exit_code = 0
        try:
            # Use shlex to avoid shell=True where possible.
            # We fall back to shell=True only when the command contains
            # pipes, redirects, or other shell metacharacters.
            has_shell_meta = any(c in cmd for c in ("|", ">", "<", "&", ";", "$", "`"))
            if has_shell_meta:
                proc = subprocess.run(cmd, shell=True)
            else:
                proc = subprocess.run(shlex.split(cmd))
            exit_code = proc.returncode
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
            exit_code = 130

        runtime = round(time.perf_counter() - t0, 3)

        # --- Log the execution ---
        cur = conn.execute(
            """
            INSERT INTO logs
                (project_id, timestamp, category, content, cmd,
                 tool_version, git_hash, runtime_s, env_info, exit_code)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                p_id,
                datetime.now().isoformat(),
                "RUN",
                f"Executed: {cmd[:120]}",
                cmd,
                ver,
                sha,
                runtime,
                json.dumps(env),
                exit_code,
            ),
        )
        log_id = cur.lastrowid

        # --- Track outputs and wire lineage ---
        missing_outputs: List[str] = []
        for f in outputs:
            if not Path(f).exists():
                missing_outputs.append(f)
                console.print(f"[yellow]Warning:[/yellow] Output '{f}' not found after run.")
                continue
            out_id = upsert_file(conn, p_id, f)
            for i_id in in_ids:
                conn.execute(
                    "INSERT INTO lineage (log_id, input_file_id, output_file_id) VALUES (?,?,?)",
                    (log_id, i_id, out_id),
                )
            # Also record outputs with no inputs (e.g. download steps)
            if not in_ids:
                conn.execute(
                    "INSERT INTO lineage (log_id, input_file_id, output_file_id) VALUES (?,?,?)",
                    (log_id, None, out_id),
                )

    status = "[green]OK[/green]" if exit_code == 0 else f"[red]FAILED (exit {exit_code})[/red]"
    console.print(f"Done in {runtime}s — {status}")
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def monitor(ctx: typer.Context):
    """
    Run a command while sampling peak RAM and average CPU every 0.5 s.

    Example:
        biln monitor "samtools sort big.bam -o sorted.bam"
    """
    cmd = " ".join(ctx.args)
    if not cmd:
        console.print("[red]Error:[/red] No command provided.")
        raise typer.Exit(code=1)

    console.print(f"[bold]Monitoring:[/bold] {cmd}")

    has_shell_meta = any(c in cmd for c in ("|", ">", "<", "&", ";", "$", "`"))
    proc = subprocess.Popen(
        cmd if has_shell_meta else shlex.split(cmd),
        shell=has_shell_meta,
    )

    peak_mem_mb: float = 0.0
    cpu_samples: List[float] = []
    t0 = time.perf_counter()

    try:
        while proc.poll() is None:
            try:
                ps = psutil.Process(proc.pid)
                mem = ps.memory_info().rss / (1024 ** 2)
                cpu = ps.cpu_percent(interval=0.1)
                if mem > peak_mem_mb:
                    peak_mem_mb = mem
                cpu_samples.append(cpu)
            except psutil.NoSuchProcess:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        proc.kill()

    runtime = round(time.perf_counter() - t0, 3)
    avg_cpu  = round(sum(cpu_samples) / len(cpu_samples), 2) if cpu_samples else 0.0

    metrics = {
        "peak_ram_mb": round(peak_mem_mb, 2),
        "avg_cpu_pct": avg_cpu,
        "runtime_s":   runtime,
    }

    with get_db() as conn:
        p_id, _ = require_project(conn)
        conn.execute(
            """
            INSERT INTO logs (project_id, timestamp, category, content, cmd, runtime_s)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (p_id, datetime.now().isoformat(), "MONITOR", json.dumps(metrics), cmd, runtime),
        )

    console.print(
        Panel(
            f"Peak RAM : [bold]{metrics['peak_ram_mb']} MB[/bold]\n"
            f"Avg CPU  : [bold]{metrics['avg_cpu_pct']}%[/bold]\n"
            f"Runtime  : [bold]{runtime}s[/bold]",
            title="Resource Audit",
        )
    )


@app.command()
def replay(
    log_id:  int,
    dry_run: bool = typer.Option(False, "--dry-run", help="Print command without running it"),
):
    """Re-run a specific command from history by its log ID."""
    with get_db() as conn:
        row = conn.execute("SELECT cmd FROM logs WHERE id = ?", (log_id,)).fetchone()

    if not row or not row["cmd"]:
        console.print(f"[red]Log ID {log_id} not found or has no command.[/red]")
        raise typer.Exit(code=1)

    console.print(Panel(row["cmd"], title=f"Command from log #{log_id}"))

    if dry_run:
        return

    if Confirm.ask("Execute now?"):
        subprocess.run(row["cmd"], shell=True)


# ---------------------------------------------------------------------------
# QUERYING
# ---------------------------------------------------------------------------

@app.command()
def history(limit: int = typer.Option(HISTORY_LIMIT, help="Number of rows to show")):
    """Show recent command history for the active project."""
    with get_db() as conn:
        p_id, p_name = require_project(conn)
        rows = conn.execute(
            """
            SELECT id, timestamp, category, cmd, content, exit_code
            FROM logs WHERE project_id = ?
            ORDER BY id DESC LIMIT ?
            """,
            (p_id, limit),
        ).fetchall()

    if not rows:
        console.print("[yellow]No history yet.[/yellow]")
        return

    table = Table(title=f"History — {p_name}")
    table.add_column("ID",       style="dim", justify="right")
    table.add_column("Time",     style="cyan")
    table.add_column("Category", style="magenta")
    table.add_column("Detail")
    table.add_column("Status",   justify="center")

    for r in rows:
        detail = (r["cmd"] or r["content"] or "")[:60]
        if len(r["cmd"] or r["content"] or "") > 60:
            detail += "…"
        code   = r["exit_code"]
        status = (
            "[green]✔[/green]"   if code == 0
            else f"[red]✘ {code}[/red]" if code is not None
            else ""
        )
        table.add_row(
            str(r["id"]),
            r["timestamp"][11:16],
            r["category"],
            detail,
            status,
        )
    console.print(table)


@app.command()
def search(query: str):
    """Full-text search across commands and notes."""
    with get_db() as conn:
        p_id, _ = require_project(conn)
        rows = conn.execute(
            """
            SELECT id, timestamp, category, cmd, content
            FROM logs
            WHERE project_id = ?
              AND (content LIKE ? OR cmd LIKE ?)
            ORDER BY id DESC
            """,
            (p_id, f"%{query}%", f"%{query}%"),
        ).fetchall()

    if not rows:
        console.print(f"[yellow]No results for '{query}'.[/yellow]")
        return

    table = Table(title=f"Search: '{query}'")
    table.add_column("ID",   style="dim", justify="right")
    table.add_column("Date", style="cyan")
    table.add_column("Category")
    table.add_column("Match")

    for r in rows:
        match = (r["cmd"] or r["content"] or "")[:70]
        table.add_row(str(r["id"]), r["timestamp"][:10], r["category"], match)
    console.print(table)


@app.command()
def lineage(path: str):
    """Trace the provenance chain for a file (what produced it and from what)."""
    abs_path = str(Path(path).resolve())

    with get_db() as conn:
        # What produced this file?
        produced = conn.execute(
            """
            SELECT l.id AS log_id, l.cmd, l.timestamp,
                   f_in.path AS src
            FROM lineage lin
            JOIN logs  l    ON lin.log_id        = l.id
            JOIN files f_in ON lin.input_file_id  = f_in.id
            JOIN files f_out ON lin.output_file_id = f_out.id
            WHERE f_out.path = ?
            """,
            (abs_path,),
        ).fetchall()

        # What did this file go on to produce?
        consumed = conn.execute(
            """
            SELECT l.id AS log_id, l.cmd, f_out.path AS dest
            FROM lineage lin
            JOIN logs  l     ON lin.log_id         = l.id
            JOIN files f_in  ON lin.input_file_id   = f_in.id
            JOIN files f_out ON lin.output_file_id  = f_out.id
            WHERE f_in.path = ?
            """,
            (abs_path,),
        ).fetchall()

    name = Path(abs_path).name

    if not produced and not consumed:
        console.print(f"[yellow]No lineage found for '{name}'.[/yellow]")
        return

    console.print(Panel(f"[bold]{name}[/bold]", title="Lineage"))

    if produced:
        console.print("\n[green]Produced from:[/green]")
        for r in produced:
            console.print(f"  ← [cyan]{Path(r['src']).name}[/cyan]  via run #{r['log_id']}  ({r['timestamp'][:10]})")

    if consumed:
        console.print("\n[blue]Used to produce:[/blue]")
        for r in consumed:
            console.print(f"  → [cyan]{Path(r['dest']).name}[/cyan]  via run #{r['log_id']}")


@app.command()
def compare(id1: int, id2: int):
    """
    Compare two runs: command, runtime, tool version, git hash,
    and the MD5 hashes of their output files.
    """
    with get_db() as conn:
        r1 = conn.execute("SELECT * FROM logs WHERE id = ?", (id1,)).fetchone()
        r2 = conn.execute("SELECT * FROM logs WHERE id = ?", (id2,)).fetchone()

        if not r1 or not r2:
            console.print("[red]One or both log IDs not found.[/red]")
            raise typer.Exit(code=1)

        out1 = conn.execute(
            "SELECT f.path, f.hash_md5 FROM files f JOIN lineage l ON f.id = l.output_file_id WHERE l.log_id = ?",
            (id1,),
        ).fetchall()
        out2 = conn.execute(
            "SELECT f.path, f.hash_md5 FROM files f JOIN lineage l ON f.id = l.output_file_id WHERE l.log_id = ?",
            (id2,),
        ).fetchall()

    table = Table(title=f"Run {id1} vs Run {id2}")
    table.add_column("Metric", style="bold")
    table.add_column(f"Run {id1}", style="cyan")
    table.add_column(f"Run {id2}", style="cyan")
    table.add_column("Match", justify="center")

    for field in ("cmd", "tool_version", "git_hash", "runtime_s"):
        v1    = str(r1[field] or "")
        v2    = str(r2[field] or "")
        match = "[green]✔[/green]" if v1 == v2 else "[red]✘[/red]"
        # Truncate long values for display
        table.add_row(field, v1[:40], v2[:40], match)

    h1 = {Path(r["path"]).name: r["hash_md5"] for r in out1}
    h2 = {Path(r["path"]).name: r["hash_md5"] for r in out2}

    for fname in sorted(set(h1) | set(h2)):
        match = "[green]identical[/green]" if h1.get(fname) == h2.get(fname) else "[red]different[/red]"
        table.add_row(f"[dim]file:[/dim] {fname}", h1.get(fname, "—")[:12], h2.get(fname, "—")[:12], match)

    console.print(table)


@app.command()
def stats():
    """Show statistics for the active project."""
    with get_db() as conn:
        p_id, p_name = require_project(conn)
        n_runs    = conn.execute("SELECT COUNT(*) FROM logs WHERE project_id = ? AND category='RUN'",     (p_id,)).fetchone()[0]
        n_ok      = conn.execute("SELECT COUNT(*) FROM logs WHERE project_id = ? AND category='RUN' AND exit_code=0", (p_id,)).fetchone()[0]
        n_files   = conn.execute("SELECT COUNT(*) FROM files  WHERE project_id = ?",                    (p_id,)).fetchone()[0]
        n_notes   = conn.execute("SELECT COUNT(*) FROM logs WHERE project_id = ? AND category='NOTE'",   (p_id,)).fetchone()[0]
        total_rt  = conn.execute("SELECT SUM(runtime_s) FROM logs WHERE project_id = ? AND runtime_s IS NOT NULL", (p_id,)).fetchone()[0] or 0.0

    rate = f"{round(n_ok/n_runs*100, 1)}%" if n_runs else "—"
    console.print(
        Panel(
            f"[bold]Project:[/bold]      {p_name}\n"
            f"[bold]Runs:[/bold]         {n_runs}  (success rate: {rate})\n"
            f"[bold]Files tracked:[/bold]{n_files}\n"
            f"[bold]Notes:[/bold]        {n_notes}\n"
            f"[bold]Total runtime:[/bold]{round(total_rt, 1)}s",
            title="Project Stats",
        )
    )


@app.command()
def show(log_id: int):
    """Open the output file(s) of a run in the system default viewer."""
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT f.path FROM files f
            JOIN lineage l ON f.id = l.output_file_id
            WHERE l.log_id = ?
            """,
            (log_id,),
        ).fetchall()

    if not rows:
        console.print(f"[red]No output files found for run #{log_id}.[/red]")
        raise typer.Exit(code=1)

    for r in rows:
        p = Path(r["path"])
        if not p.exists():
            console.print(f"[yellow]Missing:[/yellow] {r['path']}")
            continue
        system = platform.system()
        if system == "Darwin":
            subprocess.run(["open", str(p)])
        elif system == "Windows":
            os.startfile(str(p))
        else:
            subprocess.run(["xdg-open", str(p)])


# ---------------------------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------------------------

@app.command("export-snakemake")
def export_snakemake(
    filename: str = typer.Option("Snakefile", help="Output filename"),
):
    """
    Export the tracked run history as a portable Snakemake pipeline.

    Paths are converted to relative (from cwd) for portability.
    'rule all' targets only FINAL outputs (outputs never reused as inputs).
    """
    with get_db() as conn:
        p_id, p_name = require_project(conn)

        runs = conn.execute(
            """
            SELECT id, cmd, timestamp
            FROM logs
            WHERE project_id = ? AND category = 'RUN'
            ORDER BY id ASC
            """,
            (p_id,),
        ).fetchall()

        if not runs:
            console.print(f"[yellow]No runs recorded for '{p_name}'.[/yellow]")
            return

        # Pre-compute which file IDs are ever used as inputs
        all_input_ids: set = {
            r[0]
            for r in conn.execute("SELECT DISTINCT input_file_id FROM lineage WHERE input_file_id IS NOT NULL").fetchall()
        }
        all_output_ids: set = {
            r[0]
            for r in conn.execute("SELECT DISTINCT output_file_id FROM lineage WHERE output_file_id IS NOT NULL").fetchall()
        }
        final_output_ids = all_output_ids - all_input_ids

        cwd          = Path.cwd()
        rules        = []
        final_targets: List[str] = []

        for step_num, run in enumerate(runs, start=1):
            log_id = run["id"]

            inputs = conn.execute(
                """
                SELECT f.id, f.path
                FROM files f JOIN lineage l ON f.id = l.input_file_id
                WHERE l.log_id = ? AND l.input_file_id IS NOT NULL
                """,
                (log_id,),
            ).fetchall()

            outputs = conn.execute(
                """
                SELECT f.id, f.path
                FROM files f JOIN lineage l ON f.id = l.output_file_id
                WHERE l.log_id = ? AND l.output_file_id IS NOT NULL
                """,
                (log_id,),
            ).fetchall()

            if not outputs:
                console.print(
                    f"[yellow]Warning:[/yellow] Run #{log_id} has no tracked outputs — skipped."
                )
                continue

            in_paths  = [f'"{_abs_to_rel(r["path"], cwd)}"' for r in inputs]
            out_paths = [f'"{_abs_to_rel(r["path"], cwd)}"' for r in outputs]

            for r in outputs:
                if r["id"] in final_output_ids:
                    final_targets.append(f'"{_abs_to_rel(r["path"], cwd)}"')

            # Human-readable rule name: step_001_<output_stem>
            stem      = Path(outputs[0]["path"]).stem.replace("-", "_").replace(".", "_")
            rule_name = f"step_{step_num:03d}_{stem}"
            safe_cmd  = _safe_shell_cmd(run["cmd"])

            lines = [f"rule {rule_name}:"]
            if in_paths:
                lines.append(f"    input:\n        {', '.join(in_paths)}")
            lines.append(    f"    output:\n        {', '.join(out_paths)}")
            lines.append(    f"    shell:\n        '''{safe_cmd}'''")
            lines.append("")
            rules.append("\n".join(lines))

    if not rules:
        console.print("[yellow]Nothing to export — no runs had tracked outputs.[/yellow]")
        return

    unique_targets = sorted(set(final_targets))
    if not unique_targets:
        console.print(
            "[yellow]Warning:[/yellow] Could not determine final targets "
            "(all outputs are intermediate). 'rule all' omitted."
        )

    with open(filename, "w") as fh:
        fh.write(f"# Generated by BILN v2.0 — project: {p_name}\n")
        fh.write(f"# Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if unique_targets:
            fh.write("rule all:\n")
            fh.write(f"    input:\n        {', '.join(unique_targets)}\n\n\n")

        fh.write("\n".join(rules))

    console.print(
        f"[green]Snakefile written to '{filename}'[/green] "
        f"({len(rules)} rule(s), {len(unique_targets)} final target(s))."
    )


@app.command()
def snapshot():
    """Export the current Conda environment to a YAML file for reproducibility."""
    with get_db() as conn:
        p_id, name = require_project(conn)

    env_file = BILN_DIR / f"{name}_environment.yml"
    console.print(f"[yellow]Capturing Conda environment → {env_file}[/yellow]")

    try:
        subprocess.run(
            f"conda env export --no-builds > {env_file}",
            shell=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]Failed:[/red] {exc}\nIs conda in your PATH?")
        raise typer.Exit(code=1)

    with get_db() as conn:
        p_id, _ = require_project(conn)
        conn.execute(
            "INSERT INTO logs (project_id, timestamp, category, content) VALUES (?,?,?,?)",
            (p_id, datetime.now().isoformat(), "SNAPSHOT", str(env_file)),
        )

    console.print(f"[green]Snapshot saved.[/green] Run [bold]biln containerize[/bold] to generate a Dockerfile.")


@app.command()
def containerize():
    """Generate a Dockerfile that reproduces the current conda environment."""
    with get_db() as conn:
        _, name = require_project(conn)

    env_file = BILN_DIR / f"{name}_environment.yml"
    if not env_file.exists():
        console.print(
            "[red]No environment snapshot found.[/red] "
            "Run [bold]biln snapshot[/bold] first."
        )
        raise typer.Exit(code=1)

    dockerfile = Path("Dockerfile")
    dockerfile.write_text(
        f"# Generated by BILN v2.0 for project: {name}\n"
        "FROM continuumio/miniconda3\n\n"
        "WORKDIR /analysis\n\n"
        f"COPY .biln/{name}_environment.yml /tmp/env.yml\n\n"
        "RUN conda env create -f /tmp/env.yml && conda clean -afy\n\n"
        "SHELL [\"conda\", \"run\", \"-n\", \"$(head -1 /tmp/env.yml | cut -d' ' -f2)\", \"/bin/bash\", \"-c\"]\n\n"
        "CMD [\"/bin/bash\"]\n"
    )

    console.print(
        Panel(
            f"[green]Dockerfile created.[/green]\n\n"
            f"Build with:\n[cyan]docker build -t biln-{name.lower()} .[/cyan]",
            title="Containerize",
        )
    )


@app.command()
def methods(
    detailed: bool = typer.Option(False, "--detailed", help="Include every unique parameter set"),
):
    """
    Auto-generate a Methods & Materials draft for your manuscript
    based on recorded tool versions and parameters.
    """
    with get_db() as conn:
        p_id, _ = require_project(conn)
        tools = conn.execute(
            "SELECT DISTINCT tool_version, cmd FROM logs WHERE project_id = ? AND category='RUN'",
            (p_id,),
        ).fetchall()

    if not tools:
        console.print("[yellow]No runs recorded yet.[/yellow]")
        return

    software: dict = {}
    for t in tools:
        if not t["cmd"]:
            continue
        binary = shlex.split(t["cmd"])[0]
        ver    = t["tool_version"] or "unknown"
        software.setdefault(binary, {"version": ver, "params": set()})
        software[binary]["params"].add(t["cmd"])

    tool_list = ", ".join(
        f"**{name}** (v{info['version']})" for name, info in software.items()
    )

    text = (
        "### Methods & Materials (Draft)\n\n"
        "Data analysis was performed using a BILN-tracked reproducible pipeline. "
        f"The following software was used: {tool_list}. "
        f"All analyses ran on {platform.system()} and were tracked with BILN v2.0 "
        "for provenance and reproducibility."
    )

    console.print(Panel(Markdown(text), title="Methods Draft", expand=False))

    if detailed:
        extra = "\n**Detailed Command Log:**\n"
        for name, info in software.items():
            extra += f"\n- *{name}*:\n"
            for p in sorted(info["params"]):
                extra += f"  - `{p}`\n"
        console.print(Markdown(extra))


@app.command()
def cite():
    """Alias for `methods` — list tools used, formatted for citation."""
    methods()


# ---------------------------------------------------------------------------
# FILE MANAGEMENT
# ---------------------------------------------------------------------------

@app.command()
def annotate(path: str, note: str):
    """
    Attach a persistent note to a tracked file.

    Example:
        biln annotate results.vcf "High-quality variants, QUAL > 30 filter applied"
    """
    abs_path = str(Path(path).resolve())

    with get_db() as conn:
        p_id, _ = require_project(conn)

        if not conn.execute(
            "SELECT id FROM files WHERE project_id = ? AND path = ?", (p_id, abs_path)
        ).fetchone():
            if not Path(abs_path).exists():
                console.print(f"[red]File not found:[/red] {path}")
                raise typer.Exit(code=1)
            console.print(f"[dim]Tracking new file: {path}[/dim]")
            upsert_file(conn, p_id, abs_path)

        display = Path(abs_path).name
        conn.execute(
            "INSERT INTO logs (project_id, timestamp, category, content) VALUES (?,?,?,?)",
            (p_id, datetime.now().isoformat(), "ANNOTATION", f"[{display}] {note}"),
        )

    console.print(
        Panel(
            f"[bold cyan]File:[/bold cyan] {Path(abs_path).name}\n"
            f"[bold cyan]Note:[/bold cyan] {note}",
            title="Annotation Saved",
        )
    )


@app.command()
def verify():
    """
    Re-hash all tracked files and report any that have changed
    since they were last recorded.
    """
    with get_db() as conn:
        p_id, _ = require_project(conn)
        rows = conn.execute(
            "SELECT path, hash_md5 FROM files WHERE project_id = ?", (p_id,)
        ).fetchall()

    if not rows:
        console.print("[yellow]No files tracked yet.[/yellow]")
        return

    ok = changed = missing = 0
    for r in rows:
        current = file_hash_md5(r["path"])
        if current == "missing":
            console.print(f"[yellow]MISSING[/yellow]  {r['path']}")
            missing += 1
        elif current == r["hash_md5"]:
            console.print(f"[green]OK[/green]       {r['path']}")
            ok += 1
        else:
            console.print(f"[red]CHANGED[/red]  {r['path']}")
            changed += 1

    console.print(
        f"\nSummary: [green]{ok} OK[/green]  "
        f"[red]{changed} changed[/red]  "
        f"[yellow]{missing} missing[/yellow]"
    )


@app.command()
def archive(
    dry_run: bool = typer.Option(False, "--dry-run", help="List candidates without moving"),
):
    """
    Move intermediate files (output of one step and input of another)
    larger than {LARGE_FILE_MB} MB to cold storage in .biln/archive/.
    """
    with get_db() as conn:
        p_id, _ = require_project(conn)
        candidates = conn.execute(
            """
            SELECT id, path FROM files
            WHERE project_id   = ?
              AND archived     = 0
              AND id IN (SELECT output_file_id FROM lineage WHERE output_file_id IS NOT NULL)
              AND id IN (SELECT input_file_id  FROM lineage WHERE input_file_id  IS NOT NULL)
            """,
            (p_id,),
        ).fetchall()

        to_move = [
            f for f in candidates
            if Path(f["path"]).exists()
            and Path(f["path"]).stat().st_size > LARGE_FILE_MB * 1024 * 1024
        ]

        if not to_move:
            console.print("[green]No large intermediate files to archive.[/green]")
            return

        table = Table(title="Archive Candidates")
        table.add_column("Path")
        table.add_column("Size (MB)", justify="right")
        for f in to_move:
            mb = round(Path(f["path"]).stat().st_size / (1024 ** 2), 1)
            table.add_row(f["path"], str(mb))
        console.print(table)

        if dry_run:
            return

        if not Confirm.ask("Archive these files?"):
            return

        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        for f in track(to_move, description="Moving…"):
            dest = ARCHIVE_DIR / Path(f["path"]).name
            shutil.move(f["path"], dest)
            conn.execute(
                "UPDATE files SET archived = 1, path = ? WHERE id = ?",
                (str(dest.resolve()), f["id"]),
            )


@app.command()
def samples(csv_file: str):
    """
    Import sample metadata from a CSV file.

    Required columns: filename, sample, condition
    Optional column:  replicate
    """
    try:
        import pandas as pd
    except ImportError:
        console.print("[red]pandas is required. Run: pip install pandas[/red]")
        raise typer.Exit(code=1)

    try:
        df = pd.read_csv(csv_file)
    except Exception as exc:
        console.print(f"[red]Could not read CSV:[/red] {exc}")
        raise typer.Exit(code=1)

    required = {"filename", "sample", "condition"}
    missing  = required - set(df.columns)
    if missing:
        console.print(f"[red]Missing columns:[/red] {', '.join(missing)}")
        raise typer.Exit(code=1)

    with get_db() as conn:
        p_id, _ = require_project(conn)
        count = 0
        for _, row in df.iterrows():
            conn.execute(
                """
                INSERT INTO samples (project_id, sample_name, condition, replicate, file_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    p_id,
                    row["sample"],
                    row["condition"],
                    row.get("replicate", ""),
                    row["filename"],
                ),
            )
            count += 1

    console.print(f"[green]Imported {count} sample(s).[/green]")


# ---------------------------------------------------------------------------
# MAINTENANCE & DIAGNOSTICS
# ---------------------------------------------------------------------------

@app.command()
def doctor():
    """Run a sanity check on the project: disk space, missing files, DB health."""
    with get_db() as conn:
        p_id, name = require_project(conn)
        total_files   = conn.execute("SELECT COUNT(*) FROM files  WHERE project_id = ?", (p_id,)).fetchone()[0]
        missing_files = sum(
            1 for r in conn.execute("SELECT path FROM files WHERE project_id = ?", (p_id,))
            if not Path(r["path"]).exists()
        )
        schema_ver = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()

    _, _, free = shutil.disk_usage("/")
    free_gb = round(free / (1024 ** 3), 1)

    issues = []
    if missing_files:
        issues.append(f"[red]{missing_files} file(s) missing from disk[/red]")
    if free_gb < 10:
        issues.append(f"[yellow]Low disk space: {free_gb} GB free[/yellow]")

    status_str = "\n".join(issues) if issues else "[green]All checks passed.[/green]"

    console.print(
        Panel(
            f"[bold]Project:[/bold]       {name}\n"
            f"[bold]DB schema:[/bold]     v{schema_ver['value'] if schema_ver else '?'}\n"
            f"[bold]Files tracked:[/bold] {total_files}\n"
            f"[bold]Missing:[/bold]       {missing_files}\n"
            f"[bold]Disk free:[/bold]     {free_gb} GB\n\n"
            + status_str,
            title="BILN Doctor",
        )
    )


@app.command()
def system():
    """Log current system hardware specification."""
    specs = {
        "os":      platform.system(),
        "version": platform.version(),
        "cpu":     platform.processor(),
        "cores":   os.cpu_count(),
        "ram_gb":  round(psutil.virtual_memory().total / (1024 ** 3), 1),
    }

    with get_db() as conn:
        p_id, _ = require_project(conn)
        conn.execute(
            "INSERT INTO logs (project_id, timestamp, category, content) VALUES (?,?,?,?)",
            (p_id, datetime.now().isoformat(), "SYSTEM", json.dumps(specs)),
        )

    console.print(Panel(
        "\n".join(f"[bold]{k}:[/bold] {v}" for k, v in specs.items()),
        title="System Info Logged",
    ))


# ---------------------------------------------------------------------------
# REPORTING & EXPORT
# ---------------------------------------------------------------------------

@app.command()
def export(output: str = typer.Option("PROVENANCE.md", help="Output markdown file")):
    """
    Write a full Markdown provenance document for the active project
    covering every logged event, command, and file lineage.
    """
    with get_db() as conn:
        p_id, p_name = require_project(conn)
        logs = conn.execute(
            "SELECT * FROM logs WHERE project_id = ? ORDER BY timestamp ASC", (p_id,)
        ).fetchall()

        # Fetch all lineage records up-front while connection is open
        all_lineage: dict = {}
        for entry in logs:
            rows = conn.execute(
                """
                SELECT
                    (SELECT path FROM files WHERE id = lin.input_file_id)  AS in_path,
                    (SELECT path FROM files WHERE id = lin.output_file_id) AS out_path
                FROM lineage lin WHERE log_id = ?
                """,
                (entry["id"],),
            ).fetchall()
            all_lineage[entry["id"]] = rows

    with open(output, "w") as fh:
        fh.write(f"# Provenance: {p_name}\n\n")
        fh.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        fh.write(f"**Events:**    {len(logs)}\n\n---\n\n")

        for entry in logs:
            fh.write(f"### {entry['timestamp'][:16]}  |  {entry['category']}\n\n")
            cat = entry["category"]

            if cat == "RUN":
                fh.write(f"```bash\n{entry['cmd']}\n```\n\n")
                fh.write(f"| Field | Value |\n|---|---|\n")
                fh.write(f"| Tool version | `{entry['tool_version']}` |\n")
                fh.write(f"| Git hash     | `{entry['git_hash']}` |\n")
                fh.write(f"| Runtime      | {entry['runtime_s']}s |\n")
                fh.write(f"| Exit code    | `{entry['exit_code']}` |\n\n")

                lin = all_lineage.get(entry["id"], [])
                ins  = sorted({Path(r["in_path"]).name  for r in lin if r["in_path"]})
                outs = sorted({Path(r["out_path"]).name for r in lin if r["out_path"]})
                if ins:
                    fh.write(f"**Inputs:** {', '.join(f'`{i}`' for i in ins)}  \n")
                if outs:
                    fh.write(f"**Outputs:** {', '.join(f'`{o}`' for o in outs)}  \n")

            elif cat in ("NOTE", "ANNOTATION"):
                fh.write(f"> {entry['content']}\n")

            elif cat == "MONITOR":
                try:
                    m = json.loads(entry["content"])
                    fh.write(
                        f"- Peak RAM: **{m.get('peak_ram_mb', '?')} MB**\n"
                        f"- Avg CPU:  **{m.get('avg_cpu_pct', '?')}%**\n"
                        f"- Runtime:  **{m.get('runtime_s', '?')}s**\n"
                    )
                except (json.JSONDecodeError, TypeError):
                    fh.write(f"{entry['content']}\n")

            elif cat == "SYSTEM":
                try:
                    s = json.loads(entry["content"])
                    fh.write("\n".join(f"- **{k}:** {v}" for k, v in s.items()) + "\n")
                except (json.JSONDecodeError, TypeError):
                    fh.write(f"{entry['content']}\n")

            else:
                fh.write(f"{entry['content'] or ''}\n")

            fh.write("\n---\n\n")

    console.print(
        Panel(f"[green]Provenance written to:[/green] [bold]{output}[/bold]", title="Export Complete")
    )


@app.command()
def viz(output: str = typer.Option("workflow.dot", help="Output Graphviz DOT file")):
    """
    Generate a Graphviz DOT file of the full workflow lineage.

    Render with:
        dot -Tpng workflow.dot -o workflow.png
    """
    with get_db() as conn:
        p_id, p_name = require_project(conn)
        links = conn.execute(
            """
            SELECT f_in.path AS src, f_out.path AS dest, log.cmd
            FROM lineage lin
            JOIN logs  log   ON lin.log_id         = log.id
            JOIN files f_in  ON lin.input_file_id   = f_in.id
            JOIN files f_out ON lin.output_file_id  = f_out.id
            WHERE log.project_id = ?
            LIMIT ?
            """,
            (p_id, MAX_LINEAGE_VIZ),
        ).fetchall()

    if not links:
        console.print("[yellow]No lineage to visualize.[/yellow]")
        return

    with open(output, "w") as fh:
        fh.write(f'digraph "{p_name}" {{\n')
        fh.write('    rankdir="LR";\n')
        fh.write('    node [shape=box, style="filled,rounded", fillcolor="#E8F0FE", fontname="Arial"];\n')
        fh.write('    edge [fontname="Verdana", fontsize=9];\n\n')

        for lnk in links:
            src  = Path(lnk["src"]).name
            dest = Path(lnk["dest"]).name
            tool = shlex.split(lnk["cmd"])[0] if lnk["cmd"] else "?"
            fh.write(f'    "{src}" -> "{dest}" [label=" {tool} "];\n')

        fh.write("}\n")

    console.print(
        Panel(
            f"[green]DOT file saved:[/green] [bold]{output}[/bold]\n\n"
            f"Render with:\n[cyan]dot -Tpng {output} -o workflow.png[/cyan]\n"
            "Or paste into [link=https://dreampuf.github.io/GraphvizOnline/]Graphviz Online[/link]",
            title="Lineage Graph",
        )
    )


@app.command()
def dashboard(output: Optional[str] = typer.Argument(None, help="HTML output filename")):
    """
    Generate an interactive HTML dashboard with runtime charts,
    success rates, lineage graph, and a full activity log.
    """
    if not HAS_JINJA:
        console.print("[red]jinja2 required. Run: pip install jinja2[/red]")
        raise typer.Exit(code=1)

    with get_db() as conn:
        p_id, p_name = require_project(conn)
        logs  = conn.execute(
            "SELECT * FROM logs WHERE project_id = ? ORDER BY timestamp DESC", (p_id,)
        ).fetchall()
        files = conn.execute(
            "SELECT * FROM files WHERE project_id = ?", (p_id,)
        ).fetchall()
        links = conn.execute(
            """
            SELECT f_in.path AS src, f_out.path AS dest, log.cmd
            FROM lineage lin
            JOIN logs  log   ON lin.log_id         = log.id
            JOIN files f_in  ON lin.input_file_id   = f_in.id
            JOIN files f_out ON lin.output_file_id  = f_out.id
            WHERE log.project_id = ?
            LIMIT ?
            """,
            (p_id, MAX_LINEAGE_VIZ),
        ).fetchall()

    output = output or f"{p_name}_dashboard.html"

    n_runs   = sum(1 for l in logs if l["category"] == "RUN")
    n_ok     = sum(1 for l in logs if l["category"] == "RUN" and l["exit_code"] == 0)
    n_fail   = n_runs - n_ok
    total_rt = round(sum(l["runtime_s"] for l in logs if l["runtime_s"]) or 0, 1)
    rate     = round(n_ok / n_runs * 100, 1) if n_runs else 0

    runtime_data = [
        {"time": l["timestamp"][11:16], "val": l["runtime_s"], "cmd": (l["cmd"] or "")[:30]}
        for l in reversed(logs)
        if l["category"] == "RUN" and l["runtime_s"]
    ][-20:]

    mermaid_lines = ["graph LR"]
    for lnk in links:
        src  = Path(lnk["src"]).name
        dest = Path(lnk["dest"]).name
        tool = shlex.split(lnk["cmd"])[0] if lnk["cmd"] else "?"
        mermaid_lines.append(f'    {src} -->|"{tool}"| {dest}')
    mermaid_code = "\n".join(mermaid_lines)

    # Safe JSON for embedding in JS
    rt_json     = json.dumps(runtime_data)
    log_rows    = [
        {
            "ts":       l["timestamp"][5:16],
            "category": l["category"],
            "detail":   (l["cmd"] or l["content"] or "")[:80],
            "exit":     l["exit_code"],
        }
        for l in logs
    ]
    log_json = json.dumps(log_rows)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>BILN Dashboard — {p_name}</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<style>
  body {{ background:#f5f7fa; font-family:system-ui,sans-serif; }}
  .card {{ border:none; box-shadow:0 1px 4px rgba(0,0,0,.08); }}
  .stat-num {{ font-size:2rem; font-weight:700; }}
  code {{ background:#f0f0f0; padding:1px 4px; border-radius:3px; font-size:.85em; }}
</style>
</head>
<body>
<nav class="navbar navbar-dark bg-dark mb-4 px-3">
  <span class="navbar-brand fw-bold">BILN Dashboard</span>
  <span class="badge bg-primary fs-6">{p_name}</span>
</nav>
<div class="container-fluid px-4">

  <!-- Stats row -->
  <div class="row g-3 mb-4">
    <div class="col-6 col-md-3">
      <div class="card p-3 border-start border-primary border-4">
        <div class="text-muted small text-uppercase fw-bold">Runs</div>
        <div class="stat-num text-primary">{n_runs}</div>
      </div>
    </div>
    <div class="col-6 col-md-3">
      <div class="card p-3 border-start border-success border-4">
        <div class="text-muted small text-uppercase fw-bold">Files Tracked</div>
        <div class="stat-num text-success">{len(files)}</div>
      </div>
    </div>
    <div class="col-6 col-md-3">
      <div class="card p-3 border-start border-info border-4">
        <div class="text-muted small text-uppercase fw-bold">Total Runtime</div>
        <div class="stat-num text-info">{total_rt}s</div>
      </div>
    </div>
    <div class="col-6 col-md-3">
      <div class="card p-3 border-start border-warning border-4">
        <div class="text-muted small text-uppercase fw-bold">Success Rate</div>
        <div class="stat-num text-warning">{rate}%</div>
      </div>
    </div>
  </div>

  <!-- Charts row -->
  <div class="row g-3 mb-4">
    <div class="col-md-8">
      <div class="card p-3 h-100">
        <h6 class="text-muted">Runtime per Run (last 20)</h6>
        <div id="rtChart"></div>
      </div>
    </div>
    <div class="col-md-4">
      <div class="card p-3 h-100">
        <h6 class="text-muted">Success / Failure</h6>
        <div id="pieChart"></div>
      </div>
    </div>
  </div>

  <!-- Lineage -->
  <div class="row g-3 mb-4">
    <div class="col-12">
      <div class="card p-3">
        <h6 class="text-muted">Workflow Lineage</h6>
        <div class="mermaid">{mermaid_code}</div>
      </div>
    </div>
  </div>

  <!-- Log table -->
  <div class="row g-3 mb-5">
    <div class="col-12">
      <div class="card p-3">
        <h6 class="text-muted">Activity Log</h6>
        <div id="logTable"></div>
      </div>
    </div>
  </div>

</div>
<script>
const rtData = {rt_json};
Plotly.newPlot('rtChart',[{{
  x: rtData.map(d=>d.time), y: rtData.map(d=>d.val),
  type:'bar', marker:{{color:'#4e73df'}},
  text: rtData.map(d=>d.cmd), hoverinfo:'text+y'
}}],{{margin:{{t:10,b:40,l:40,r:10}},height:280,yaxis:{{title:'seconds'}}}});

Plotly.newPlot('pieChart',[{{
  values:[{n_ok},{n_fail}], labels:['Success','Fail'],
  type:'pie', hole:.45, marker:{{colors:['#1cc88a','#e74a3b']}}
}}],{{margin:{{t:10,b:10,l:10,r:10}},height:280,showlegend:true}});

const logs = {log_json};
const rows = logs.map(l=>{{
  const badge = l.category==='RUN' ? 'bg-primary' : l.category==='MONITOR' ? 'bg-info' : 'bg-secondary';
  const stat  = l.exit===0 ? '<span class="text-success">✔</span>'
               : l.exit!=null ? `<span class="text-danger">✘ (${{l.exit}})</span>` : '';
  return `<tr>
    <td class="small text-muted">${{l.ts}}</td>
    <td><span class="badge ${{badge}}">${{l.category}}</span></td>
    <td><code>${{l.detail.replace(/</g,'&lt;')}}</code></td>
    <td>${{stat}}</td>
  </tr>`;
}}).join('');

document.getElementById('logTable').innerHTML =
  '<table class="table table-sm table-hover">' +
  '<thead class="table-light"><tr><th>Time</th><th>Type</th><th>Detail</th><th>Status</th></tr></thead>' +
  '<tbody>' + rows + '</tbody></table>';

mermaid.initialize({{startOnLoad:true, theme:'neutral'}});
</script>
</body>
</html>"""

    Path(output).write_text(html)
    console.print(
        Panel(
            f"[green]Dashboard written to:[/green] [bold]{output}[/bold]",
            title="Dashboard Ready",
        )
    )


@app.command()
def publish(name: str = typer.Option("Research_Bundle", help="Bundle name prefix")):
    """
    Bundle the database, provenance doc, and lineage graph into a ZIP for sharing.
    """
    with get_db() as conn:
        _, p_name = require_project(conn)

    md_file  = Path("PROVENANCE.md")
    dot_file = Path("workflow.dot")

    # Refresh docs
    export(output=str(md_file))
    viz(output=str(dot_file))

    stamp      = datetime.now().strftime("%Y%m%d")
    bundle_stem = f"{name}_{p_name}_{stamp}"
    tmp_dir    = Path(f"_tmp_{bundle_stem}")
    tmp_dir.mkdir(exist_ok=True)

    try:
        shutil.copy(DB_PATH, tmp_dir / "biln.db")
        if md_file.exists():
            shutil.copy(md_file, tmp_dir / "PROVENANCE.md")
        if dot_file.exists():
            shutil.copy(dot_file, tmp_dir / "workflow.dot")

        zip_path = shutil.make_archive(bundle_stem, "zip", tmp_dir)
        console.print(f"[green]Bundle created:[/green] [bold]{zip_path}[/bold]")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# MISCELLANEOUS
# ---------------------------------------------------------------------------

@app.command()
def hello():
    """Greet the user."""
    console.print(
        Panel(
            "[bold green]Welcome to BILN v2.0![/bold green]\n\n"
            "Run [bold]biln --help[/bold] to see all commands.\n"
            "Run [bold]biln manual[/bold] for the full reference.",
            title="Hello",
        )
    )


@app.command()
def manual():
    """Print the full BILN command reference."""
    md = """
# BILN v2.0 — Command Reference

## Project Management
| Command | Description |
|---|---|
| `init` | Initialise database in current directory |
| `project <name>` | Switch active project (use `--create` to make a new one) |
| `list-projects` | List all projects |
| `hello` | Welcome message |

## Execution & Logging
| Command | Description |
|---|---|
| `run -i <in> -o <out> "cmd"` | Run a command with full provenance tracking |
| `monitor "cmd"` | Run a command and sample peak RAM / avg CPU |
| `log <message>` | Record a free-text note |
| `replay <id>` | Re-run a command from history (use `--dry-run` to preview) |

## Querying
| Command | Description |
|---|---|
| `history` | Recent logs (use `--limit N`) |
| `search <term>` | Full-text search across commands and notes |
| `lineage <file>` | Trace what produced a file and what it produced |
| `compare <id1> <id2>` | Diff two runs by metadata and output hashes |
| `stats` | Project-level statistics |
| `show <id>` | Open output file(s) of a run |

## Files & Metadata
| Command | Description |
|---|---|
| `annotate <file> <note>` | Attach a note to a tracked file |
| `verify` | Re-hash all tracked files and report changes |
| `archive` | Move large intermediate files to cold storage |
| `samples <csv>` | Import sample metadata (columns: filename, sample, condition) |

## Reproducibility
| Command | Description |
|---|---|
| `export-snakemake` | Generate a portable Snakefile from run history |
| `snapshot` | Export current Conda environment to YAML |
| `containerize` | Generate a Dockerfile from the environment snapshot |
| `methods` | Draft a Methods & Materials paragraph |
| `cite` | Alias for `methods` |

## Reporting
| Command | Description |
|---|---|
| `export` | Write full provenance to Markdown |
| `viz` | Generate a Graphviz DOT lineage graph |
| `dashboard` | Generate an interactive HTML dashboard |
| `publish` | Bundle database + docs into a ZIP for sharing |

## Maintenance
| Command | Description |
|---|---|
| `doctor` | Sanity check: disk space, missing files, DB health |
| `system` | Log system hardware specs |
| `manual` | This reference |
"""
    console.print(Markdown(md))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()