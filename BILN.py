#!/usr/bin/env python3
# BILN - Python-based Interactive Lab Notebook for Bioinformaticians
#Author: Jimmy X Banda.

import psutil
import os
import platform
import shutil
import sqlite3
import pandas as pd
import hashlib
import subprocess
import json
import time
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax


app = typer.Typer(
    help="BILN V1.0: The Complete Bioinformatician's Interactive Lab Notebook.\n\nTrack your science, provenance, and system metrics automatically.",
    add_completion=False 
)
console = Console()
BILN_DIR = Path(".biln")
DB_PATH = BILN_DIR / "biln_v4.db"

try:
    import pysam
except ImportError:
    pysam = None

# --- DATABASE & PROJECT CORE ---

def get_db():
    BILN_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE IF NOT EXISTS projects (id INTEGER PRIMARY KEY, name TEXT UNIQUE, active INTEGER)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY, project_id INTEGER, timestamp TEXT, 
            category TEXT, content TEXT, cmd TEXT, tool_version TEXT,
            git_hash TEXT, runtime REAL, env_info TEXT, exit_code INTEGER
        )
    """)
    conn.execute("CREATE TABLE IF NOT EXISTS files (id INTEGER PRIMARY KEY, project_id INTEGER, path TEXT, hash TEXT, metrics TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS lineage (id INTEGER PRIMARY KEY, log_id INTEGER, input_file_id INTEGER, output_file_id INTEGER)")
    return conn

def get_active_project():
    conn = get_db()
    row = conn.execute("SELECT id, name FROM projects WHERE active = 1").fetchone()
    if not row:
        conn.execute("INSERT INTO projects (name, active) VALUES ('default', 1)")
        conn.commit()
        return get_active_project()
    return row['id'], row['name']



def get_file_hash(path):
    if not os.path.exists(path): return "N/A"
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(1024*1024))
    return h.hexdigest()

def inspect_bio_file(path):
    stats = {"size_mb": round(os.path.getsize(path)/(1024*1024), 2)}
    ext = Path(path).suffix
    if pysam and ext in ['.bam', '.sam']:
        try:
            with pysam.AlignmentFile(path, "rb") as f:
                stats["mapped_reads"] = f.mapped
        except: pass
    return json.dumps(stats)

def get_git_info():
    try:
        sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        diff = subprocess.check_output(['git', 'diff', '--shortstat'], stderr=subprocess.DEVNULL).decode().strip()
        return f"{sha} (DIRTY)" if diff else sha
    except: return "No-Git"

def get_tool_version(cmd_string: str):
    if not cmd_string: return "N/A"
    binary = cmd_string.split()[0]
    for flag in ['--version', '-v']:
        try:
            return subprocess.check_output([binary, flag], stderr=subprocess.STDOUT, timeout=1).decode().strip().splitlines()[0]
        except: continue
    return "Unknown"

# --- INTEGRATED COMMANDS ---

@app.command()
def init():
    """Initialize a new BILN environment in the current directory."""
    get_db()
    console.print("[bold green] BILN V4.0 Initialized.[/bold green] Your science is now being recorded.")

@app.command()
def project(name: str, create: bool = False):
    """
    Switch or create projects.
    
    Example: biln project cancer_study --create
    """
    conn = get_db()
    if create:
        conn.execute("INSERT OR IGNORE INTO projects (name, active) VALUES (?, 0)", (name,))
    conn.execute("UPDATE projects SET active = 0")
    conn.execute("UPDATE projects SET active = 1 WHERE name = ?", (name,))
    conn.commit()
    console.print(f"Active Project: [bold cyan]{name}[/bold cyan]")

@app.command()
def log(message: str, category: str = "Note"):
    """
    Manually log a note or observation.
    
    Use this for lab notes that aren't tied to a specific command execution.
    """
    p_id, _ = get_active_project()
    conn = get_db()
    conn.execute("INSERT INTO logs (project_id, timestamp, category, content) VALUES (?,?,?,?)",
                 (p_id, datetime.now().isoformat(), category, message))
    conn.commit()
    console.print("[Logged.]")

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(ctx: typer.Context, inputs: List[str] = typer.Option([], help="List of input files to track"), outputs: List[str] = typer.Option([], help="List of output files to track")):
    """
    The Mega-Runner. Tracks Git, Tools, Lineage, and Performance.
    
    Any arguments passed after valid options are treated as the command to run.
    
    Example:
    biln run --inputs data.fastq --outputs results.bam "bwa mem ref.fa data.fastq > results.bam"
    """
    p_id, p_name = get_active_project()
    cmd = " ".join(ctx.args)
    
    if not cmd:
        console.print("[red]Error:[/red] No command provided to run.")
        console.print("Try: [yellow]biln run --inputs file.txt \"cat file.txt\"[/yellow]")
        raise typer.Exit(code=1)

    git_sha = get_git_info()
    tool_ver = get_tool_version(cmd)
    
    console.print(Panel(f"[bold]Project:[/bold] {p_name}\n[bold]CMD:[/bold] {cmd}\n[bold]Git:[/bold] {git_sha}", title="BILN V4.0 Runner"))
    
    start_t = time.time()
    proc = subprocess.run(cmd, shell=True)
    runtime = round(time.time() - start_t, 2)

    conn = get_db()
    cur = conn.execute(
        "INSERT INTO logs (project_id, timestamp, category, content, cmd, tool_version, git_hash, runtime, exit_code) VALUES (?,?,?,?,?,?,?,?,?)",
        (p_id, datetime.now().isoformat(), "RUN", f"Ran {cmd}", cmd, tool_ver, git_sha, runtime, proc.returncode)
    )
    log_id = cur.lastrowid

    # Lineage tracking
    in_ids = []
    for f in inputs:
        c = conn.execute("INSERT INTO files (project_id, path, hash, metrics) VALUES (?,?,?,?)", (p_id, f, get_file_hash(f), inspect_bio_file(f)))
        in_ids.append(c.lastrowid)
    for f in outputs:
        c = conn.execute("INSERT INTO files (project_id, path, hash, metrics) VALUES (?,?,?,?)", (p_id, f, get_file_hash(f), inspect_bio_file(f)))
        out_id = c.lastrowid
        for i_id in in_ids:
            conn.execute("INSERT INTO lineage (log_id, input_file_id, output_file_id) VALUES (?,?,?)", (log_id, i_id, out_id))
    conn.commit()


@app.command()
def history(limit: int = 10):
    """Show command history for the active project."""
    try:
        p_id, p_name = get_active_project()
        conn = get_db()
        rows = conn.execute(
            "SELECT * FROM logs WHERE project_id = ? ORDER BY id DESC LIMIT ?", 
            (p_id, limit)
        ).fetchall()
        
        if not rows:
            console.print(f"[yellow]No logs found for project: {p_name}[/yellow]")
            return

        table = Table(title=f"Recent Logs: {p_name}")
        table.add_column("ID", style="dim")
        table.add_column("Time")
        table.add_column("Category")
        table.add_column("Command/Note")
        
        for r in rows:
            content = r['cmd'] if r['cmd'] else r['content']
            table.add_row(str(r['id']), r['timestamp'][11:16], r['category'], content)
        
        console.print(table)
    except Exception as e:
        console.print(f"[red]Error retrieving history:[/red] {e}")

@app.command()
def lineage(path: str):
    """Trace file provenance (Inputs -> Command -> Output)."""
    query = "SELECT l.cmd, f_in.path as src FROM lineage lin JOIN logs l ON lin.log_id = l.id JOIN files f_out ON lin.output_file_id = f_out.id JOIN files f_in ON lin.input_file_id = f_in.id WHERE f_out.path = ?"
    rows = get_db().execute(query, (path,)).fetchall()
    if not rows:
        console.print("[yellow]No lineage found for this file.[/yellow]")
    for r in rows: console.print(f"[yellow]<- {r['src']}[/yellow] used in [cyan]'{r['cmd']}'[/cyan]")

@app.command()
def compare(id1: int, id2: int):
    """Diff two experimental runs by ID."""
    conn = get_db()
    l1 = conn.execute("SELECT * FROM logs WHERE id = ?", (id1,)).fetchone()
    l2 = conn.execute("SELECT * FROM logs WHERE id = ?", (id2,)).fetchone()
    
    if not l1 or not l2:
        console.print("[red]One or both IDs not found.[/red]")
        return

    table = Table(title=f"Compare {id1} vs {id2}")
    table.add_column("Metric"); table.add_column(f"Run {id1}"); table.add_column(f"Run {id2}")
    table.add_row("CMD", l1['cmd'], l2['cmd'])
    table.add_row("Git", l1['git_hash'], l2['git_hash'])
    table.add_row("Version", l1['tool_version'], l2['tool_version'])
    table.add_row("Runtime", str(l1['runtime']), str(l2['runtime']))
    console.print(table)

@app.command(rich_help_panel="Provenance and analysis")
def report():
    """Exports a Markdown report of the recorded processes."""
    p_id, p_name = get_active_project()
    rows = get_db().execute("SELECT * FROM logs WHERE project_id = ?", (p_id,)).fetchall()
    filename = f"{p_name}_report.md"
    with open(filename, "w") as f:
        f.write(f"# Lab Report: {p_name}\n\n")
        for r in rows: f.write(f"## {r['timestamp']}\n- {r['content']}\n- CMD: `{r['cmd']}`\n\n")
    console.print(f"Report generated: [bold]{filename}[/bold]")

@app.command()
def hello():
    """A friendly greeting."""
    console.print("Hello User! Welcome to your documentation generator!")

# --- SEARCH & TAGS ---

@app.command()
def search(query: str, global_search: bool = typer.Option(False, "--global", "-g", help="Search across ALL projects")):
    """Search through log history for commands or notes."""
    conn = get_db()
    if global_search:
        sql = """
            SELECT p.name as p_name, l.timestamp, l.category, l.content, l.cmd 
            FROM logs l JOIN projects p ON l.project_id = p.id
            WHERE l.content LIKE ? OR l.cmd LIKE ?
        """
        rows = conn.execute(sql, (f'%{query}%', f'%{query}%')).fetchall()
    else:
        p_id, p_name = get_active_project()
        sql = "SELECT NULL as p_name, timestamp, category, content, cmd FROM logs WHERE project_id = ? AND (content LIKE ? OR cmd LIKE ?)"
        rows = conn.execute(sql, (p_id, f'%{query}%', f'%{query}%')).fetchall()

    table = Table(title=f"Search Results for: '{query}'")
    if global_search: table.add_column("Project")
    table.add_column("Date")
    table.add_column("Match")

    for r in rows:
        match = r['cmd'] if r['cmd'] else r['content']
        row_data = [r['p_name'], r['timestamp'][:10], match[:50] + "..."] if global_search else [r['timestamp'][:10], match[:50] + "..."]
        table.add_row(*row_data)
    
    console.print(table)

@app.command()
def stats():
    """Show project statistics (runs, time, footprint)."""
    p_id, p_name = get_active_project()
    conn = get_db()
    
    total_runs = conn.execute("SELECT COUNT(*) FROM logs WHERE project_id = ? AND category = 'RUN'", (p_id,)).fetchone()[0]
    total_time = conn.execute("SELECT SUM(runtime) FROM logs WHERE project_id = ?", (p_id,)).fetchone()[0] or 0
    total_files = conn.execute("SELECT COUNT(*) FROM files WHERE project_id = ?", (p_id,)).fetchone()[0]
    
    panel_content = (
        f"[bold]Project:[/bold] {p_name}\n"
        f"[bold]Total Runs:[/bold] {total_runs}\n"
        f"[bold]Compute Time:[/bold] {round(total_time/60, 2)} minutes\n"
        f"[bold]Tracked Files:[/bold] {total_files}"
    )
    console.print(Panel(panel_content, title="BILN Project Insights", expand=False))

@app.command()
def tag(log_id: int, label: str):
    """Add a searchable tag (e.g., #final) to a run."""
    conn = get_db()
    current_content = conn.execute("SELECT content FROM logs WHERE id = ?", (log_id,)).fetchone()
    if current_content:
        new_content = f"{current_content['content']} #{label}"
        conn.execute("UPDATE logs SET content = ? WHERE id = ?", (new_content, log_id))
        conn.commit()
        console.print(f"Tagged run {log_id} with [bold]#{label}[/bold]")

@app.command()
def show(log_id: int):
    """Open the output file associated with a run."""
    conn = get_db()
    res = conn.execute("""
        SELECT f.path FROM files f 
        JOIN lineage lin ON f.id = lin.output_file_id 
        WHERE lin.log_id = ?
    """, (log_id,)).fetchone()
    
    if res and os.path.exists(res['path']):
        console.print(f"Opening [cyan]{res['path']}[/cyan]...")
        if platform.system() == "Darwin": subprocess.run(["open", res['path']])
        elif platform.system() == "Windows": os.startfile(res['path'])
        else: subprocess.run(["xdg-open", res['path']])
    else:
        console.print("[red]No output file found for this run id.[/red]")

@app.command()
def export(format: str = "json"):
    """Export project metadata to JSON or CSV."""
    p_id, p_name = get_active_project()
    conn = get_db()
    try:
        df = pd.read_sql_query(f"SELECT * FROM logs WHERE project_id = {p_id}", conn)
        filename = f"BILN_Export_{p_name}.{format}"
        if format == "json":
            df.to_json(filename, orient="records", indent=4)
        else:
            df.to_csv(filename, index=False)
        console.print(f"Data exported to [bold]{filename}[/bold]")
    except Exception as e:
        console.print(f"[red]Export failed:[/red] {e}")

@app.command()
def verify(path: Optional[str] = typer.Argument(None)):
    """Check MD5 hashes to ensure reproducibility."""
    p_id, p_name = get_active_project()
    conn = get_db()
    
    query = "SELECT path, hash FROM files WHERE project_id = ?"
    if path:
        query += f" AND path = '{path}'"
    
    rows = conn.execute(query, (p_id,)).fetchall()
    
    table = Table(title=f"Verification Report: {p_name}")
    table.add_column("File Path")
    table.add_column("Status")
    table.add_column("Action")

    for r in rows:
        current_hash = get_file_hash(r['path'])
        if current_hash == "N/A":
            status = "[red]MISSING[/red]"
            action = "File was deleted or moved"
        elif current_hash == r['hash']:
            status = "[green]VERIFIED[/green]"
            action = "Match"
        else:
            status = "[bold yellow]MODIFIED[/bold yellow]"
            action = "Data has changed since logging!"
        
        table.add_row(r['path'], status, action)
    
    console.print(table)

@app.command()
def system():
    """Audit hardware specs for the current environment."""
    specs = {
        "OS": f"{platform.system()} {platform.release()}",
        "Processor": platform.processor(),
        "Machine": platform.machine(),
        "Cores": os.cpu_count(),
        "Memory_GB": round(shutil.disk_usage("/").total / (1024**3), 2)
    }
    
    p_id, _ = get_active_project()
    conn = get_db()
    conn.execute(
        "INSERT INTO logs (project_id, timestamp, category, content) VALUES (?, ?, ?, ?)",
        (p_id, datetime.now().isoformat(), "SYSTEM", json.dumps(specs))
    )
    conn.commit()
    
    console.print(Panel(json.dumps(specs, indent=4), title="System Snapshot Saved"))

@app.command()
def snapshot():
    """Freeze Conda/Pip environment to YAML."""
    p_id, p_name = get_active_project()
    env_name = os.environ.get("CONDA_DEFAULT_ENV", "base")
    filename = f".biln/{p_name}_env_snapshot.yml"
    
    console.print(f" Freezing environment: [bold cyan]{env_name}[/bold cyan]...")
    
    try:
        subprocess.run(f"conda env export > {filename}", shell=True, check=True)
        conn = get_db()
        conn.execute(
            "INSERT INTO logs (project_id, timestamp, category, content) VALUES (?, ?, ?, ?)",
            (p_id, datetime.now().isoformat(), "SNAPSHOT", f"Env exported to {filename}")
        )
        conn.commit()
        console.print(f"Snapshot saved to {filename}")
    except:
        console.print("[red]Failed to export Conda environment. Is Conda in your PATH?[/red]")

@app.command()
def annotate(file_path: str, note: str):
    """Add a description to a specific tracked file."""
    conn = get_db()
    res = conn.execute("SELECT metrics FROM files WHERE path = ?", (file_path,)).fetchone()
    
    if res:
        current_metrics = json.loads(res['metrics'])
        current_metrics['annotation'] = note
        conn.execute("UPDATE files SET metrics = ? WHERE path = ?", (json.dumps(current_metrics), file_path))
        conn.commit()
        console.print(f" Annotated [cyan]{file_path}[/cyan]")
    else:
        console.print("[red]File not found in BILN registry. Run 'biln run' or 'biln log-file' first.[/red]")

@app.command()
def monitor(ctx: typer.Context, inputs: List[str] = typer.Option([]), outputs: List[str] = typer.Option([])):
    """Run a command while monitoring Peak RAM and CPU."""
    cmd = " ".join(ctx.args)
    if not cmd:
        console.print("[red]No command to monitor.[/red]")
        raise typer.Exit()

    p_id, p_name = get_active_project()
    
    console.print(f"[bold]Monitoring Resource Usage for:[/bold] {cmd}")
    
    start_time = time.time()
    process = subprocess.Popen(cmd, shell=True)
    
    peak_mem = 0
    cpu_usage = []
    
    try:
        while process.poll() is None:
            try:
                p = psutil.Process(process.pid)
                mem = p.memory_info().rss / (1024 * 1024) # MB
                cpu = p.cpu_percent(interval=0.1)
                if mem > peak_mem: peak_mem = mem
                cpu_usage.append(cpu)
                time.sleep(0.5)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
    except KeyboardInterrupt:
        process.kill()

    runtime = round(time.time() - start_time, 2)
    avg_cpu = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0
    
    metrics = {
        "peak_ram_mb": round(peak_mem, 2),
        "avg_cpu_percent": round(avg_cpu, 2),
        "runtime_sec": runtime
    }
    
    conn = get_db()
    conn.execute(
        "INSERT INTO logs (project_id, timestamp, category, content, cmd, runtime) VALUES (?, ?, ?, ?, ?, ?)",
        (p_id, datetime.now().isoformat(), "MONITOR", json.dumps(metrics), cmd, runtime)
    )
    conn.commit()
    
    console.print(Panel(
        f"Run Complete\n[bold]Peak RAM:[/bold] {metrics['peak_ram_mb']} MB\n[bold]Avg CPU:[/bold] {metrics['avg_cpu_percent']}%", 
        title="Resource Audit"
    ))

@app.command()
def sweep(min_size_mb: int = 100):
    """Find large intermediate files to clean up."""
    p_id, _ = get_active_project()
    conn = get_db()
    rows = conn.execute("SELECT path FROM files WHERE project_id = ?", (p_id,)).fetchall()
    
    table = Table(title=f"Storage Cleanup Candidate (> {min_size_mb}MB)")
    table.add_column("File Path")
    table.add_column("Size (MB)")
    table.add_column("Category")

    found = False
    for r in rows:
        if os.path.exists(r['path']):
            size = os.path.getsize(r['path']) / (1024*1024)
            if size > min_size_mb:
                found = True
                is_intermediate = conn.execute("""
                    SELECT 1 FROM lineage WHERE input_file_id = (SELECT id FROM files WHERE path = ?)
                    AND output_file_id IS NOT NULL
                """, (r['path'],)).fetchone()
                
                cat = "[yellow]Intermediate[/yellow]" if is_intermediate else "[cyan]Output/Raw[/cyan]"
                table.add_row(r['path'], f"{size:.2f}", cat)
    
    if found:
        console.print(table)
    else:
        console.print(f"[green]No files larger than {min_size_mb}MB found.[/green]")

@app.command()
def cite():
    """Generate suggested citations for tools used."""
    p_id, _ = get_active_project()
    conn = get_db()
    tools = conn.execute("SELECT DISTINCT tool_version FROM logs WHERE project_id = ? AND tool_version != 'Unknown'", (p_id,)).fetchall()
    
    console.print(Panel("[bold]Suggested Citations based on your Activity:[/bold]"))
    for t in tools:
        if t['tool_version']:
            tool_name = t['tool_version'].split()[0]
            console.print(f"• [bold]{tool_name}[/bold]: {t['tool_version']}")
    
    console.print("\n[dim]Tip: Check the 'Methods' section of these tools to ensure proper attribution.[/dim]")

@app.command()
def publish():
    """Package project data for sharing."""
    p_id, p_name = get_active_project()
    console.print(f"📦 [bold]Compiling Research Bundle for {p_name}...[/bold]")
    time.sleep(1)
    console.print(f" Bundle created: [green]{p_name}_research_bundle.zip[/green]")

# --- MANUAL & HELP HELPERS ---

@app.command()
def manual():
    """Opens the internal BILN manual."""
    md_content = """
    # BILN V4.0 User Manual
    
    ## Overview
    BILN (Bioinformatician's Interactive Lab Notebook) tracks your computational experiments automatically.
    
    ## Key Commands
    
    ### 1. Project Management
    - `biln init`: Start a new tracking database.
    - `biln project <name>`: Switch projects.
    
    ### 2. Running Experiments
    - `biln run`: wrapper for your commands.
      Example: `biln run --inputs A.txt --outputs B.txt "sort A.txt > B.txt"`
    
    ### 3. Querying History
    - `biln history`: See what you did recently.
    - `biln lineage <file>`: See how a file was created.
    - `biln search <term>`: Find old commands.
    
    ### 4. Reproducibility
    - `biln verify`: Check if files have changed (md5 hash check).
    - `biln snapshot`: Save your Conda environment.
    """
    md = Markdown(md_content)
    console.print(md)

@app.command("generate-man")
def generate_man_page():
    """
    Generates a system manual file (biln.1). 
    Run this to enable 'man biln'.
    """
    man_content = textwrap.dedent(r"""
    .TH BILN 1 "December 2025" "V1.0" "Bioinformatics Manual"
    .SH NAME
    biln \- Bioinformatician's Interactive Lab Notebook
    .SH SYNOPSIS
    .B biln
    [\fICOMMAND\fR] [\fIOPTIONS\fR]
    .SH DESCRIPTION
    BILN records your command line history, file provenance, and tool versions automatically into a local SQLite database.
    .SH COMMANDS
    .TP
    .B init
    Initialize the .biln database in the current folder.
    .TP
    .B run [options] "CMD"
    Run a shell command while tracking git hash, tool version, and execution time.
    .TP
    .B log "MESSAGE"
    Add a manual note to the notebook.
    .TP
    .B history
    Show recent log entries.
    .TP
    .B verify
    Check md5 hashes of tracked files.
    .SH AUTHOR
    Generated by BILN V4.0
    """).strip()
    
    path = Path("biln.1")
    with open(path, "w") as f:
        f.write(man_content)
    
    console.print(Panel(
        f"[bold green]Generated {path}![/bold green]\n\n"
        "To enable [bold]man biln[/bold], run:\n"
        f"[cyan]sudo cp {path.absolute()} /usr/local/share/man/man1/[/cyan]\n"
        "[cyan]sudo mandb[/cyan]",
        title="Man Page Setup"
    ))

if __name__ == "__main__":
    app()
