#!/usr/bin/env python3
"""
BILN - Bioinformatician's Interactive Lab Notebook
Author: Jimmy X Banda
Version: 2.1 (2026)

A provenance-tracking CLI for bioinformatics workflows.
Tracks commands, files, lineage, environments, and resources.
Supports exporting reproducible pipelines to Snakemake and Nextflow DSL2.
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
from rich.panel import Panel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BILN_DIR        = Path(".biln")
DB_PATH         = BILN_DIR / "biln.db"
ARCHIVE_DIR     = BILN_DIR / "archive"
HASH_CHUNK_SIZE = 8 * 1024 * 1024   # 8 MB chunks for streaming hash
MAX_LINEAGE_VIZ = 50                # Max edges in mermaid graphs
DB_SCHEMA_VER   = 2                 # Schema version indicator

# Optional heavy bio dependencies
try:
    import pysam
except ImportError:
    pysam = None

# ---------------------------------------------------------------------------
# App & Console Initialisation
# ---------------------------------------------------------------------------

app = typer.Typer(
    help=(
        "BILN v2.1 — The Bioinformatician's Interactive Lab Notebook.\n\n"
        "Track commands, files, lineage, and environments reproducibly."
    ),
    add_completion=False,
    no_args_is_help=True,
)
console = Console()

# ---------------------------------------------------------------------------
# Database Architecture & Schema
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
    Context manager yielding a thread-safe sqlite3 connection with a 60-second
    lock timeout to prevent concurrent access crashes during workflow runs.
    """
    BILN_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)

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
    """Return (project_id, project_name) for the currently active project."""
    row = conn.execute(
        "SELECT id, name FROM projects WHERE active = 1"
    ).fetchone()

    if row is None:
        conn.execute(
            "INSERT OR IGNORE INTO projects (name, active) VALUES ('default', 1)"
        )
        conn.commit()
        row = conn.execute(
            "SELECT id, name FROM projects WHERE active = 1"
        ).fetchone()

    if row is None:
        console.print("[red]Error:[/red] Active project could not be loaded or created.")
        raise typer.Exit(code=1)

    return int(row["id"]), str(row["name"])


# ---------------------------------------------------------------------------
# File Utilities & Bioinformatics Metric Processors
# ---------------------------------------------------------------------------

def file_hash_md5(path: str) -> str:
    """Compute MD5 hash over a file in streaming 8MB chunks."""
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
    """Collect lightweight bioinformatics metrics without memory exhaustion."""
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
        try:
            opener = (
                __import__("gzip").open if str(p).endswith(".gz")
                else open
            )
            n = 0
            with opener(path, "rt") as fh:
                for n, _ in enumerate(fh, start=1):
                    pass
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
    """Track a file (insert or update). Returns the internal row ID."""
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
# Environment & Command Parsing Helpers
# ---------------------------------------------------------------------------

def git_info() -> str:
    """Return short Git commit SHA (+ DIRTY flag) or 'no-git'."""
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
    """Extract software version by probing common CLI version flags."""
    if not cmd_string:
        return "unknown"
    try:
        binary = shlex.split(cmd_string)[0]
    except Exception:
        return "unknown"

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
    """Capture environment execution flags."""
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


def _abs_to_rel(abs_path: str, cwd: Path) -> str:
    """Safely convert absolute path to relative string if within working directory."""
    try:
        return str(Path(abs_path).relative_to(cwd))
    except ValueError:
        return abs_path


def _safe_snakemake_cmd(cmd: str) -> str:
    """Escape backslashes, quotes, and double curly braces for Snakemake string blocks."""
    escaped_braces = cmd.replace("{", "{{").replace("}", "}}")
    return escaped_braces.replace("\\", "\\\\").replace("'''", r"\'\'\'")


def _safe_nextflow_cmd(cmd: str) -> str:
    """Escape backslashes, quotes, and dollar signs for Nextflow DSL2 script blocks."""
    cmd = cmd.replace("\\", "\\\\")
    cmd = cmd.replace("$", "\\$")
    return cmd.replace('"""', r'\"\"\"')


# ---------------------------------------------------------------------------
# CLI Commands: Core Execution & Project Switching
# ---------------------------------------------------------------------------

@app.command()
def init():
    """Initialise a new BILN environment in the current directory."""
    with get_db() as conn:
        require_project(conn)
    console.print(
        Panel(
            f"[green]BILN v2.1 initialised.[/green]\n"
            f"Database: [dim]{DB_PATH}[/dim]",
            title="BILN Ready",
        )
    )


@app.command()
def project(
    name: str,
    create: bool = typer.Option(False, "--create", "-c", help="Create project if missing"),
):
    """Switch active working project."""
    with get_db() as conn:
        exists = conn.execute(
            "SELECT id FROM projects WHERE name = ?", (name,)
        ).fetchone()

        if not exists and not create:
            console.print(
                f"[red]Project '{name}' not found.[/red] Use [bold]--create[/bold] to construct it."
            )
            raise typer.Exit(code=1)

        if not exists:
            conn.execute(
                "INSERT INTO projects (name, active) VALUES (?, 0)", (name,)
            )

        conn.execute("UPDATE projects SET active = 0")
        conn.execute("UPDATE projects SET active = 1 WHERE name = ?", (name,))

    console.print(f"Active project → [bold cyan]{name}[/bold cyan]")


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(
    ctx: typer.Context,
    inputs:  List[str] = typer.Option([], "--input",  "-i", help="Input files to track"),
    outputs: List[str] = typer.Option([], "--output", "-o", help="Output files to track"),
):
    """
    Run a CLI tool while tracking Git hash, software version, environment, runtime, and lineage.
    """
    cmd = " ".join(ctx.args)
    if not cmd:
        console.print("[red]Error:[/red] No command provided after flags.")
        raise typer.Exit(code=1)

    with get_db() as conn:
        p_id, p_name = require_project(conn)

        in_ids: List[int] = []
        for f in inputs:
            if not Path(f).exists():
                console.print(f"[yellow]Warning:[/yellow] Input file '{f}' does not exist on disk.")
            in_ids.append(upsert_file(conn, p_id, f))

        sha = git_info()
        ver = tool_version(cmd)
        env = env_snapshot()

        console.print(
            Panel(
                f"[bold]Project:[/bold] {p_name}\n"
                f"[bold]CMD:[/bold]     {cmd}\n"
                f"[bold]Env:[/bold]     {env['conda_env'] or 'base'} ({env['container']})\n"
                f"[bold]Git:[/bold]     {sha}",
                title="BILN Runner",
            )
        )

        t0 = time.perf_counter()
        exit_code = 0
        try:
            has_shell_meta = any(c in cmd for c in ("|", ">", "<", "&", ";", "$", "`"))
            if has_shell_meta:
                proc = subprocess.run(cmd, shell=True)
            else:
                proc = subprocess.run(shlex.split(cmd))
            exit_code = proc.returncode
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user.[/yellow]")
            exit_code = 130

        runtime = round(time.perf_counter() - t0, 3)

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

        for f in outputs:
            if not Path(f).exists():
                console.print(f"[yellow]Warning:[/yellow] Output '{f}' not found after execution.")
                continue
            out_id = upsert_file(conn, p_id, f)
            for i_id in in_ids:
                conn.execute(
                    "INSERT INTO lineage (log_id, input_file_id, output_file_id) VALUES (?,?,?)",
                    (log_id, i_id, out_id),
                )
            if not in_ids:
                conn.execute(
                    "INSERT INTO lineage (log_id, input_file_id, output_file_id) VALUES (?,?,?)",
                    (log_id, None, out_id),
                )

    status = "[green]OK[/green]" if exit_code == 0 else f"[red]FAILED (exit {exit_code})[/red]"
    console.print(f"Finished in {runtime}s — {status}")
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


# ---------------------------------------------------------------------------
# CLI Commands: Pipeline Exporters & Graph Generator
# ---------------------------------------------------------------------------

@app.command("export-snakemake")
def export_snakemake(
    filename: str = typer.Option("Snakefile", help="Output Snakefile name"),
):
    """Export active project history into a Snakemake workflow."""
    with get_db() as conn:
        p_id, p_name = require_project(conn)

        runs = conn.execute(
            """
            SELECT id, cmd, timestamp
            FROM logs
            WHERE project_id = ? AND category = 'RUN' AND exit_code = 0
            ORDER BY id ASC
            """,
            (p_id,),
        ).fetchall()

        if not runs:
            console.print(f"[yellow]No successful runs found for project '{p_name}'.[/yellow]")
            return

        all_input_ids: set = {
            r[0] for r in conn.execute(
                """
                SELECT DISTINCT lin.input_file_id 
                FROM lineage lin 
                JOIN logs l ON lin.log_id = l.id 
                WHERE l.project_id = ? AND lin.input_file_id IS NOT NULL
                """,
                (p_id,),
            ).fetchall()
        }

        all_output_ids: set = {
            r[0] for r in conn.execute(
                """
                SELECT DISTINCT lin.output_file_id 
                FROM lineage lin 
                JOIN logs l ON lin.log_id = l.id 
                WHERE l.project_id = ? AND lin.output_file_id IS NOT NULL
                """,
                (p_id,),
            ).fetchall()
        }

        final_output_ids = all_output_ids - all_input_ids

        cwd = Path.cwd()
        rules = []
        final_targets: List[str] = []

        for step_num, run in enumerate(runs, start=1):
            log_id = run["id"]

            inputs = conn.execute(
                """
                SELECT f.id, f.path FROM files f
                JOIN lineage l ON f.id = l.input_file_id
                WHERE l.log_id = ? AND l.input_file_id IS NOT NULL
                """,
                (log_id,),
            ).fetchall()

            outputs = conn.execute(
                """
                SELECT f.id, f.path FROM files f
                JOIN lineage l ON f.id = l.output_file_id
                WHERE l.log_id = ? AND l.output_file_id IS NOT NULL
                """,
                (log_id,),
            ).fetchall()

            if not outputs:
                continue

            in_paths  = [f'"{_abs_to_rel(r["path"], cwd)}"' for r in inputs]
            out_paths = [f'"{_abs_to_rel(r["path"], cwd)}"' for r in outputs]

            for r in outputs:
                if r["id"] in final_output_ids:
                    final_targets.append(f'"{_abs_to_rel(r["path"], cwd)}"')

            stem      = Path(outputs[0]["path"]).stem.replace("-", "_").replace(".", "_")
            rule_name = f"step_{step_num:03d}_{stem}"
            safe_cmd  = _safe_snakemake_cmd(run["cmd"])

            lines = [f"rule {rule_name}:"]
            if in_paths:
                lines.append(f"    input:\n        {', '.join(in_paths)}")
            lines.append(f"    output:\n        {', '.join(out_paths)}")
            lines.append(f"    shell:\n        '''{safe_cmd}'''")

            rules.append("\n".join(lines))

        unique_targets = list(dict.fromkeys(final_targets))

        snakefile_content = [
            f"# Generated automatically by BILN v2.1 on {datetime.now().isoformat()}",
            f"# Project: {p_name}\n",
            "rule all:",
            f"    input:\n        {', '.join(unique_targets)}\n",
            "\n\n".join(rules)
        ]

        Path(filename).write_text("\n\n".join(snakefile_content))
        console.print(f"[green]Successfully exported Snakemake workflow to [bold]{filename}[/bold][/green]")


@app.command("export-nextflow")
def export_nextflow(
    filename: str = typer.Option("main.nf", help="Output Nextflow filename"),
):
    """Export active project history into a Nextflow DSL2 workflow (main.nf)."""
    with get_db() as conn:
        p_id, p_name = require_project(conn)

        runs = conn.execute(
            """
            SELECT id, cmd, timestamp
            FROM logs
            WHERE project_id = ? AND category = 'RUN' AND exit_code = 0
            ORDER BY id ASC
            """,
            (p_id,),
        ).fetchall()

        if not runs:
            console.print(f"[yellow]No successful runs found for project '{p_name}'.[/yellow]")
            return

        cwd = Path.cwd()
        processes = []
        workflow_calls = []

        for step_num, run in enumerate(runs, start=1):
            log_id = run["id"]

            inputs = conn.execute(
                """
                SELECT f.path FROM files f
                JOIN lineage l ON f.id = l.input_file_id
                WHERE l.log_id = ? AND l.input_file_id IS NOT NULL
                """,
                (log_id,),
            ).fetchall()

            outputs = conn.execute(
                """
                SELECT f.path FROM files f
                JOIN lineage l ON f.id = l.output_file_id
                WHERE l.log_id = ? AND l.output_file_id IS NOT NULL
                """,
                (log_id,),
            ).fetchall()

            if not outputs:
                continue

            stem = Path(outputs[0]["path"]).stem.replace("-", "_").replace(".", "_")
            proc_name = f"STEP_{step_num:03d}_{stem.upper()}"
            safe_cmd = _safe_nextflow_cmd(run["cmd"])

            in_declarations = []
            in_args = []
            for idx, r in enumerate(inputs, start=1):
                var_name = f"in_file_{idx}"
                in_declarations.append(f"    path {var_name}")
                in_args.append(f'file("{_abs_to_rel(r["path"], cwd)}")')

            out_declarations = []
            for idx, r in enumerate(outputs, start=1):
                rel_p = _abs_to_rel(r["path"], cwd)
                out_declarations.append(f'    path "{rel_p}"')

            proc_lines = [f"process {proc_name} {{"]
            if in_declarations:
                proc_lines.append("    input:")
                proc_lines.extend(in_declarations)

            if out_declarations:
                proc_lines.append("    output:")
                proc_lines.extend(out_declarations)

            proc_lines.append("    script:")
            proc_lines.append(f'    """\n    {safe_cmd}\n    """')
            proc_lines.append("}")

            processes.append("\n".join(proc_lines))

            if in_args:
                args_str = ", ".join(in_args)
                workflow_calls.append(f"    {proc_name}({args_str})")
            else:
                workflow_calls.append(f"    {proc_name}()")

        nf_script = [
            f"// Generated automatically by BILN v2.1 on {datetime.now().isoformat()}",
            f"// Project: {p_name}\n",
            "nextflow.enable.dsl=2\n",
            "\n\n".join(processes),
            "\nworkflow {",
            "\n".join(workflow_calls),
            "}"
        ]

        Path(filename).write_text("\n".join(nf_script))
        console.print(
            f"[green]Successfully exported Nextflow DSL2 pipeline to [bold]{filename}[/bold][/green]"
        )


@app.command()
def graph(output: str = typer.Option("lineage.mmd", help="Mermaid diagram output file")):
    """Export the active workflow lineage as a visual Mermaid DAG graph."""
    with get_db() as conn:
        p_id, _ = require_project(conn)
        edges = conn.execute(
            """
            SELECT DISTINCT f_in.path AS src, f_out.path AS dest, l.cmd
            FROM lineage lin
            JOIN files f_in  ON lin.input_file_id  = f_in.id
            JOIN files f_out ON lin.output_file_id = f_out.id
            JOIN logs l      ON lin.log_id         = l.id
            WHERE l.project_id = ?
            LIMIT ?
            """,
            (p_id, MAX_LINEAGE_VIZ),
        ).fetchall()

    if not edges:
        console.print("[yellow]No project lineage found to generate graph.[/yellow]")
        return

    mermaid_lines = ["graph TD"]
    for e in edges:
        src = Path(e["src"]).name
        dest = Path(e["dest"]).name
        cmd_stub = e["cmd"].split()[0] if e["cmd"] else "run"
        mermaid_lines.append(f'    {src}["{src}"] -->|"{cmd_stub}"| {dest}["{dest}"]')

    Path(output).write_text("\n".join(mermaid_lines))
    console.print(f"[green]Saved Mermaid DAG to [bold]{output}[/bold][/green]")


if __name__ == "__main__":
    app()
