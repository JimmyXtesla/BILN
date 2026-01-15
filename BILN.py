#!/usr/bin/env python3
# BILN - Python-based Interactive Lab Notebook for Bioinformaticians
# Author: Jimmy X Banda.
# Version: 1.0 (2025) - Updated File Tracking

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
from rich.prompt import Confirm
from rich.progress import track

# Optional Imports for advanced features
try:
    import pysam
except ImportError:
    pysam = None

try:
    from jinja2 import Template
except ImportError:
    Template = None

app = typer.Typer(
    help="BILN V1.0: The Bioinformatician's Interactive Lab Notebook.\n\nFrom interactive exploration to reproducible pipelines.",
    add_completion=False 
)
console = Console()
BILN_DIR = Path(".biln")
DB_PATH = BILN_DIR / "biln_v1.db"

# --- DATABASE & CORE FUNCTIONS ---

def get_db():
    BILN_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # Projects
    conn.execute("CREATE TABLE IF NOT EXISTS projects (id INTEGER PRIMARY KEY, name TEXT UNIQUE, active INTEGER)")
    
    # Logs (Events/Runs)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY, project_id INTEGER, timestamp TEXT, 
            category TEXT, content TEXT, cmd TEXT, tool_version TEXT,
            git_hash TEXT, runtime REAL, env_info TEXT, exit_code INTEGER
        )
    """)
    
    # Files (Tracked Data) - Added UNIQUE constraint to prevent duplicates
    conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY, project_id INTEGER, path TEXT, 
            hash TEXT, metrics TEXT, archived INTEGER DEFAULT 0,
            UNIQUE(project_id, path)
        )
    """)
    
    # Lineage (Provenance)
    conn.execute("CREATE TABLE IF NOT EXISTS lineage (id INTEGER PRIMARY KEY, log_id INTEGER, input_file_id INTEGER, output_file_id INTEGER)")
    
    # Samples (Metadata)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY, project_id INTEGER, 
            sample_name TEXT, condition TEXT, replicate TEXT,
            file_path TEXT
        )
    """)
    
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
    p = Path(path)
    if not p.exists(): return "N/A"
    h = hashlib.md5()
    try:
        with open(p, "rb") as f:
            h.update(f.read(10 * 1024 * 1024)) 
        return h.hexdigest()
    except: return "Error"

def track_file(conn, project_id, path):
    """Helper to ensure files are tracked using absolute paths without duplication."""
    abs_path = str(Path(path).resolve())
    f_hash = get_file_hash(abs_path)
    metrics = inspect_bio_file(abs_path)
    
    # UPSERT logic: Insert new file record or update existing one if path matches
    conn.execute("""
        INSERT INTO files (project_id, path, hash, metrics) VALUES (?, ?, ?, ?)
        ON CONFLICT(project_id, path) DO UPDATE SET 
            hash=excluded.hash, 
            metrics=excluded.metrics
    """, (project_id, abs_path, f_hash, metrics))
    
    res = conn.execute("SELECT id FROM files WHERE project_id = ? AND path = ?", (project_id, abs_path)).fetchone()
    return res['id']

def inspect_bio_file(path):
    if not os.path.exists(path): return json.dumps({"error": "File missing"})
    stats = {"size_mb": round(os.path.getsize(path)/(1024*1024), 2)}
    ext = Path(path).suffix
    if pysam and ext in ['.bam', '.sam', '.cram']:
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
    for flag in ['--version', '-v', '-V']:
        try:
            return subprocess.check_output([binary, flag], stderr=subprocess.STDOUT, timeout=1).decode().strip().splitlines()[0]
        except: continue
    return "Unknown"

# --- PROJECT MANAGEMENT ---

@app.command()
def init():
    """Initialize a new BILN environment."""
    get_db()
    console.print("[bold green]BILN V1.0 Initialized.[/bold green] Ready to track.")

@app.command()
def project(name: str, create: bool = False):
    """Switch or create projects."""
    conn = get_db()
    if create:
        conn.execute("INSERT OR IGNORE INTO projects (name, active) VALUES (?, 0)", (name,))
    conn.execute("UPDATE projects SET active = 0")
    conn.execute("UPDATE projects SET active = 1 WHERE name = ?", (name,))
    conn.commit()
    console.print(f"Active Project: [bold cyan]{name}[/bold cyan]")

@app.command()
def hello():
    """A friendly greeting."""
    console.print("Hello User! Welcome to BILN V1.0. Enjoy")

# --- EXECUTION & LOGGING ---

@app.command()
def log(message: str, category: str = "Note"):
    """Manually log a note or observation."""
    p_id, _ = get_active_project()
    conn = get_db()
    conn.execute("INSERT INTO logs (project_id, timestamp, category, content) VALUES (?,?,?,?)",
                 (p_id, datetime.now().isoformat(), category, message))
    conn.commit()
    console.print("[Logged.]")

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(ctx: typer.Context, inputs: List[str] = typer.Option([], help="Input files"), outputs: List[str] = typer.Option([], help="Output files")):
    """
    Run a command while tracking git, tools, lineage, and environment.
    Example: biln run --inputs in.bam --outputs out.vcf "bcftools call..."
    """
    p_id, p_name = get_active_project()
    cmd = " ".join(ctx.args)
    
    if not cmd:
        console.print("[red]Error:[/red] No command provided.")
        raise typer.Exit(code=1)

    conn = get_db()
    
    # 1. Track Inputs BEFORE execution
    in_ids = []
    for f in inputs:
        in_ids.append(track_file(conn, p_id, f))

    git_sha = get_git_info()
    tool_ver = get_tool_version(cmd)
    
    env_info = {
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", None),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "base"),
        "container": "Docker" if os.path.exists("/.dockerenv") else ("Singularity" if "SINGULARITY_NAME" in os.environ else "Host")
    }

    console.print(Panel(f"[bold]Project:[/bold] {p_name}\n[bold]CMD:[/bold] {cmd}\n[bold]Env:[/bold] {env_info['conda_env']} ({env_info['container']})", title="BILN V1.0 Runner"))
    
    start_t = time.time()
    try:
        proc = subprocess.run(cmd, shell=True)
        exit_code = proc.returncode
    except KeyboardInterrupt:
        console.print("\n[red]Process interrupted.[/red]")
        exit_code = 130

    runtime = round(time.time() - start_t, 2)

    # 2. Log the execution
    cur = conn.execute(
        "INSERT INTO logs (project_id, timestamp, category, content, cmd, tool_version, git_hash, runtime, env_info, exit_code) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (p_id, datetime.now().isoformat(), "RUN", f"Ran {cmd}", cmd, tool_ver, git_sha, runtime, json.dumps(env_info), exit_code)
    )
    log_id = cur.lastrowid

    # 3. Track Outputs AFTER execution and link Lineage
    for f in outputs:
        if os.path.exists(f):
            out_id = track_file(conn, p_id, f)
            for i_id in in_ids:
                conn.execute("INSERT INTO lineage (log_id, input_file_id, output_file_id) VALUES (?,?,?)", (log_id, i_id, out_id))
    
    conn.commit()

@app.command()
def monitor(ctx: typer.Context):
    """Run a command while monitoring Peak RAM and CPU."""
    cmd = " ".join(ctx.args)
    if not cmd:
        console.print("[red]No command to monitor.[/red]")
        raise typer.Exit()

    p_id, _ = get_active_project()
    console.print(f"[bold]Monitoring:[/bold] {cmd}")
    
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
            except: break
    except KeyboardInterrupt:
        process.kill()

    runtime = round(time.time() - start_time, 2)
    avg_cpu = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0
    
    metrics = {"peak_ram_mb": round(peak_mem, 2), "avg_cpu": round(avg_cpu, 2), "runtime": runtime}
    
    conn = get_db()
    conn.execute("INSERT INTO logs (project_id, timestamp, category, content, cmd, runtime) VALUES (?, ?, ?, ?, ?, ?)",
        (p_id, datetime.now().isoformat(), "MONITOR", json.dumps(metrics), cmd, runtime))
    conn.commit()
    
    console.print(Panel(f"Peak RAM: {metrics['peak_ram_mb']} MB | Avg CPU: {metrics['avg_cpu']}%", title="Resource Audit"))

@app.command()
def replay(log_id: int, dry_run: bool = typer.Option(False, "--dry-run", help="Just print the command")):
    """Re-run a specific command from history."""
    conn = get_db()
    row = conn.execute("SELECT cmd FROM logs WHERE id = ?", (log_id,)).fetchone()
    
    if not row:
        console.print(f"[red]Log ID {log_id} not found.[/red]")
        return
    
    console.print(f"[bold]Command:[/bold] [cyan]{row['cmd']}[/cyan]")
    if not dry_run and Confirm.ask("Execute now?"):
        subprocess.run(row['cmd'], shell=True)

# --- QUERYING & ANALYSIS ---

@app.command()
def history(limit: int = 15):
    """Show command history."""
    p_id, p_name = get_active_project()
    rows = get_db().execute("SELECT * FROM logs WHERE project_id = ? ORDER BY id DESC LIMIT ?", (p_id, limit)).fetchall()
    
    table = Table(title=f"History: {p_name}")
    table.add_column("ID", style="dim"); table.add_column("Time"); table.add_column("CMD / Note"); table.add_column("Status")
    for r in rows:
        status = "[green]OK[/green]" if r['exit_code'] == 0 else ("[red]ERR[/red]" if r['exit_code'] else "")
        content = r['cmd'] if r['cmd'] else r['content']
        table.add_row(str(r['id']), r['timestamp'][11:16], content[:50] + "...", status)
    console.print(table)

@app.command()
def search(query: str):
    """Search logs for a term."""
    p_id, _ = get_active_project()
    rows = get_db().execute("SELECT id, timestamp, cmd, content FROM logs WHERE project_id = ? AND (content LIKE ? OR cmd LIKE ?)", (p_id, f'%{query}%', f'%{query}%')).fetchall()
    table = Table(title=f"Search: '{query}'")
    table.add_column("ID"); table.add_column("Date"); table.add_column("Match")
    for r in rows:
        match = r['cmd'] if r['cmd'] else r['content']
        table.add_row(str(r['id']), r['timestamp'][:10], match[:60])
    console.print(table)

@app.command()
def lineage(path: str):
    """Trace file inputs and outputs."""
    abs_path = str(Path(path).resolve())
    rows = get_db().execute("""
        SELECT l.cmd, f_in.path as src, l.id as run_id 
        FROM lineage lin 
        JOIN logs l ON lin.log_id = l.id 
        JOIN files f_in ON lin.input_file_id = f_in.id 
        JOIN files f_out ON lin.output_file_id = f_out.id 
        WHERE f_out.path = ?""", (abs_path,)).fetchall()
    if not rows: console.print("[yellow]No lineage found.[/yellow]")
    for r in rows: console.print(f"[yellow]<- {r['src']}[/yellow] used in [cyan]run {r['run_id']}[/cyan]")

@app.command()
def compare(id1: int, id2: int):
    """Compare two runs side-by-side."""
    conn = get_db()
    l1 = conn.execute("SELECT * FROM logs WHERE id = ?", (id1,)).fetchone()
    l2 = conn.execute("SELECT * FROM logs WHERE id = ?", (id2,)).fetchone()
    if l1 and l2:
        table = Table(title=f"Compare {id1} vs {id2}")
        table.add_column("Metric"); table.add_column(f"Run {id1}"); table.add_column(f"Run {id2}")
        table.add_row("CMD", l1['cmd'], l2['cmd'])
        table.add_row("Runtime", str(l1['runtime']), str(l2['runtime']))
        table.add_row("Git", l1['git_hash'], l2['git_hash'])
        console.print(table)

@app.command()
def stats():
    """Project statistics."""
    p_id, p_name = get_active_project()
    conn = get_db()
    runs = conn.execute("SELECT COUNT(*) FROM logs WHERE project_id = ? AND category='RUN'", (p_id,)).fetchone()[0]
    files = conn.execute("SELECT COUNT(*) FROM files WHERE project_id = ?", (p_id,)).fetchone()[0]
    console.print(Panel(f"Project: {p_name}\nRuns: {runs}\nFiles Tracked: {files}", title="Stats"))

@app.command()
def show(log_id: int):
    """Open output file of a run."""
    res = get_db().execute("SELECT f.path FROM files f JOIN lineage l ON f.id = l.output_file_id WHERE l.log_id = ?", (log_id,)).fetchone()
    if res and os.path.exists(res['path']):
        if platform.system() == "Darwin": subprocess.run(["open", res['path']])
        elif platform.system() == "Windows": os.startfile(res['path'])
    else: console.print("[red]File not found.[/red]")

# --- REPRODUCIBILITY & IMPACT ---

@app.command("export-snakemake")
def export_snakemake(filename: str = "Snakefile"):
    """Export history as a Snakemake pipeline."""
    p_id, p_name = get_active_project()
    conn = get_db()
    runs = conn.execute("SELECT id, cmd, timestamp FROM logs WHERE project_id = ? AND category = 'RUN' ORDER BY id ASC", (p_id,)).fetchall()

    if not runs:
        console.print(f"[yellow]No runs found for {p_name}.[/yellow]")
        return

    snake_rules = []
    all_targets = set()

    for run in runs:
        log_id = run['id']
        inputs = conn.execute("SELECT f.path FROM files f JOIN lineage l ON f.id = l.input_file_id WHERE l.log_id = ?", (log_id,)).fetchall()
        outputs = conn.execute("SELECT f.path FROM files f JOIN lineage l ON f.id = l.output_file_id WHERE l.log_id = ?", (log_id,)).fetchall()
        
        out_paths = [f'"{r["path"]}"' for r in outputs]
        if not out_paths: continue
        
        in_paths = [f'"{r["path"]}"' for r in inputs]
        all_targets.update(out_paths)
        
        rule = f"rule step_{log_id}:\n    input: {', '.join(in_paths)}\n    output: {', '.join(out_paths)}\n    shell: \"{run['cmd'].replace('"', '\\"')}\"\n"
        snake_rules.append(rule)

    with open(filename, "w") as f:
        f.write(f"# Generated by BILN V1.0 for {p_name}\nrule all:\n    input: {', '.join(sorted(list(all_targets)))}\n\n")
        f.write("\n".join(snake_rules))
    console.print(f"[green]Pipeline exported to {filename}[/green]")

@app.command()
def dashboard():
    """Generate an interactive HTML dashboard."""
    if Template is None:
        console.print("[red]Error: 'jinja2' not installed.[/red]")
        return

    p_id, p_name = get_active_project()
    conn = get_db()
    logs = conn.execute("SELECT * FROM logs WHERE project_id = ? ORDER BY id DESC", (p_id,)).fetchall()
    file_count = conn.execute("SELECT COUNT(*) FROM files WHERE project_id = ?", (p_id,)).fetchone()[0]
    
    html = """
    <!DOCTYPE html><html><head><title>BILN: {{p}}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    </head><body class="bg-light p-4"><div class="container">
    <h1> BILN Project: {{p}}</h1>
    <div class="row my-4"><div class="col"><div class="card p-3 bg-primary text-white"><h3>{{n_runs}}</h3>Runs</div></div>
    <div class="col"><div class="card p-3 bg-success text-white"><h3>{{n_files}}</h3>Files</div></div></div>
    <div class="card p-3"><table class="table"><thead><tr><th>ID</th><th>Time</th><th>CMD</th></tr></thead><tbody>
    {% for l in logs %}<tr><td>{{l.id}}</td><td>{{l.timestamp[:16]}}</td><td><code>{{l.cmd or l.content}}</code></td></tr>{% endfor %}
    </tbody></table></div></div></body></html>
    """
    
    t = Template(html)
    out = t.render(p=p_name, logs=logs, n_runs=len([l for l in logs if l['category']=='RUN']), n_files=file_count)
    
    with open(f"{p_name}_dashboard.html", "w") as f: f.write(out)
    console.print(f"[green]Dashboard saved: {p_name}_dashboard.html[/green]")


@app.command()
def viz(output: str = "workflow.dot"):
    """
    Generate a Graphviz DOT file of the project lineage.
    To view: Install graphviz and run 'dot -Tpng workflow.dot -o workflow.png'
    """
    p_id, p_name = get_active_project()
    conn = get_db()
    
    # Query to link inputs and outputs via the lineage table
    query = """
        SELECT 
            f_in.path as src, 
            f_out.path as dest, 
            log.cmd 
        FROM lineage lin
        JOIN logs log ON lin.log_id = log.id
        JOIN files f_in ON lin.input_file_id = f_in.id
        JOIN files f_out ON lin.output_file_id = f_out.id
        WHERE log.project_id = ?
    """
    links = conn.execute(query, (p_id,)).fetchall()
    
    if not links:
        console.print("[yellow]No lineage links found to visualize.[/yellow]")
        return

    with open(output, "w") as f:
        f.write(f'digraph "{p_name}" {{\n')
        f.write('    rankdir="LR";\n')
        f.write('    node [shape=box, style="filled, rounded", fillcolor="#E8F0FE", fontname="Arial"];\n')
        f.write('    edge [fontname="Verdana", fontsize=10];\n\n')
        
        for l in links:
            # We use the filename (basename) for the nodes to keep the graph clean
            src_node = os.path.basename(l["src"])
            dest_node = os.path.basename(l["dest"])
            # Get the first word of the command (the tool name) as the label
            tool_name = l["cmd"].split()[0] if l["cmd"] else "unknown"
            
            f.write(f'    "{src_node}" -> "{dest_node}" [label=" {tool_name} "];\n')
        
        f.write("}\n")
    
    console.print(Panel(
        f"[green]Graphviz file saved to:[/green] [bold]{output}[/bold]\n\n"
        "To convert to an image, use:\n"
        f"[cyan]dot -Tpng {output} -o workflow.png[/cyan]\n"
        "Or paste the content into [bold][link=https://dreampuf.github.io/GraphvizOnline/]Graphviz Online[/link][/bold]",
        title="Lineage Visualization"
    ))

@app.command()
def methods():
    """Draft Methods section text."""
    p_id, _ = get_active_project()
    tools = get_db().execute("SELECT DISTINCT tool_version, cmd FROM logs WHERE project_id = ? AND category='RUN'", (p_id,)).fetchall()
    text = "Analysis performed using: " + ", ".join([f"**{t['cmd'].split()[0]}** ({t['tool_version']})" for t in tools if t['cmd']]) + "."
    console.print(Panel(Markdown(text), title="Draft Methods"))

@app.command()
def samples(csv_file: str):
    """Import sample metadata (csv: filename, sample, condition)."""
    p_id, _ = get_active_project()
    conn = get_db()
    try:
        df = pd.read_csv(csv_file)
        count = 0
        for _, r in df.iterrows():
            conn.execute("INSERT INTO samples (project_id, sample_name, condition, file_path) VALUES (?,?,?,?)", (p_id, r['sample'], r['condition'], r['filename']))
            count += 1
        conn.commit()
        console.print(f"[green]Imported {count} samples.[/green]")
    except Exception as e: console.print(f"[red]Error: {e}[/red]")

@app.command()
def archive(dry_run: bool = False):
    """Move intermediate files to cold storage."""
    p_id, _ = get_active_project()
    conn = get_db()
    # Find files that are inputs AND outputs (intermediates)
    files = conn.execute("SELECT id, path FROM files WHERE project_id = ? AND archived = 0 AND id IN (SELECT output_file_id FROM lineage) AND id IN (SELECT input_file_id FROM lineage)", (p_id,)).fetchall()
    
    archive_dir = BILN_DIR / "archive"
    archive_dir.mkdir(exist_ok=True)
    
    to_move = [f for f in files if os.path.exists(f['path']) and os.path.getsize(f['path']) > 100*1024*1024]
    
    if not to_move:
        console.print("[green]No large intermediate files found.[/green]")
        return

    table = Table(title="Archive Candidates")
    table.add_column("File")
    for f in to_move: table.add_row(f['path'])
    console.print(table)
    
    if not dry_run and Confirm.ask("Archive these files?"):
        for f in track(to_move, description="Moving..."):
            dest = archive_dir / os.path.basename(f['path'])
            shutil.move(f['path'], dest)
            conn.execute("UPDATE files SET archived = 1, path = ? WHERE id = ?", (str(dest), f['id']))
        conn.commit()

@app.command()
def containerize():
    """Generate Dockerfile from tool history."""
    p_id, _ = get_active_project()
    logs = get_db().execute("SELECT cmd FROM logs WHERE project_id = ?", (p_id,)).fetchall()
    tools = set([l['cmd'].split()[0] for l in logs if l['cmd']])
    
    content = "FROM continuumio/miniconda3\nWORKDIR /data\n" + \
              f"RUN conda install -y -c bioconda {' '.join(tools)}\n"
    with open("Dockerfile", "w") as f: f.write(content)
    console.print("[green]Dockerfile generated.[/green]")

@app.command()
def doctor():
    """System and project sanity check."""
    p_id, name = get_active_project()
    conn = get_db()
    _, _, free = shutil.disk_usage("/")
    missing = 0
    for f in conn.execute("SELECT path FROM files WHERE project_id = ?", (p_id,)):
        if not os.path.exists(f['path']): missing += 1
    
    console.print(Panel(f"Project: {name}\nDisk Free: {free//(2**30)} GB\nMissing Files: {missing}\nDB Status: OK", title="BILN Doctor"))

@app.command()
def verify():
    """Verify file hashes."""
    p_id, _ = get_active_project()
    rows = get_db().execute("SELECT path, hash FROM files WHERE project_id = ?", (p_id,)).fetchall()
    for r in rows:
        stat = "[green]OK[/green]" if get_file_hash(r['path']) == r['hash'] else "[red]FAIL[/red]"
        console.print(f"{r['path']}: {stat}")

@app.command()
def snapshot():
    """Save Conda environment."""
    p_id, name = get_active_project()
    fname = f".biln/{name}_env.yml"
    subprocess.run(f"conda env export > {fname}", shell=True)
    get_db().execute("INSERT INTO logs (project_id, timestamp, category, content) VALUES (?,?,?,?)", (p_id, datetime.now().isoformat(), "SNAPSHOT", fname))
    get_db().commit()
    console.print(f"[green]Env saved to {fname}[/green]")

@app.command()
def export(output: str = "PROVENANCE.md"):
    """
    Generate a full Markdown documentation of the project's history and provenance.
    """
    p_id, p_name = get_active_project()
    conn = get_db()
    
    # 1. Fetch Project Info
    logs = conn.execute("SELECT * FROM logs WHERE project_id = ? ORDER BY timestamp ASC", (p_id,)).fetchall()
    
    with open(output, "w") as f:
        # Header
        f.write(f"# Project Documentation: {p_name}\n")
        f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n")
        f.write(f"- **Total Events:** {len(logs)}\n")
        f.write(f"- **Active Environment:** {os.environ.get('CONDA_DEFAULT_ENV', 'N/A')}\n\n")
        
        f.write("--- \n\n")
        f.write("## Process Log\n\n")
        
        for entry in logs:
            f.write(f"### {entry['timestamp'][:16]} | {entry['category']}\n")
            
            if entry['category'] == "RUN":
                # Detail the command execution
                f.write(f"**Command:**\n```bash\n{entry['cmd']}\n```\n")
                f.write(f"- **Tool Version:** `{entry['tool_version']}`\n")
                f.write(f"- **Git Hash:** `{entry['git_hash']}`\n")
                f.write(f"- **Runtime:** {entry['runtime']}s\n")
                f.write(f"- **Exit Code:** `{entry['exit_code']}`\n")
                
                # Retrieve Inputs and Outputs for this specific RUN
                lineage_data = conn.execute("""
                    SELECT 
                        (SELECT path FROM files WHERE id = lin.input_file_id) as input_path,
                        (SELECT path FROM files WHERE id = lin.output_file_id) as output_path
                    FROM lineage lin WHERE log_id = ?
                """, (entry['id'],)).fetchall()
                
                if lineage_data:
                    inputs = sorted(list(set([os.path.basename(r['input_path']) for r in lineage_data if r['input_path']])))
                    outputs = sorted(list(set([os.path.basename(r['output_path']) for r in lineage_data if r['output_path']])))
                    
                    if inputs: f.write(f"- **Inputs:** {', '.join([f'`{i}`' for i in inputs])}\n")
                    if outputs: f.write(f"- **Outputs:** {', '.join([f'`{o}`' for o in outputs])}\n")
                
            elif entry['category'] == "Note":
                f.write(f"> **Observation:** {entry['content']}\n")

            elif entry['category'] == "ANNOTATION":
                f.write(f"#### 📝 File Annotation\n")
                f.write(f"{entry['content']}\n")
            
            elif entry['category'] == "MONITOR":
                metrics = json.loads(entry['content'])
                f.write(f"**Resource Usage:**\n")
                f.write(f"- Peak RAM: {metrics['peak_ram_mb']} MB\n")
                f.write(f"- Avg CPU: {metrics['avg_cpu']}%\n")
                
            f.write("\n---\n\n")

    console.print(Panel(f"[green]Documentation successfully exported to:[/green] [bold]{output}[/bold]", title="Export Complete"))

@app.command()
def cite():
    """Show tools used for citation."""
    methods() # Alias for methods

@app.command()
def annotate(path: str, note: str):
    """
    Add a persistent note/annotation to a specific tracked file.
    Example: biln annotate results.vcf "High quality variants, filtered with depth > 20"
    """
    p_id, _ = get_active_project()
    conn = get_db()
    
    # Resolve to absolute path to match the database
    abs_path = str(Path(path).resolve())

    # 1. Check if the file is already tracked
    file_row = conn.execute(
        "SELECT id FROM files WHERE project_id = ? AND path = ?", 
        (p_id, abs_path)
    ).fetchone()

    if not file_row:
        # If the file isn't tracked yet, track it now so the annotation has a target
        if os.path.exists(abs_path):
            console.print(f"[dim]File not previously tracked. Adding {path} to project...[/dim]")
            track_file(conn, p_id, abs_path)
        else:
            console.print(f"[red]Error:[/red] File '{path}' does not exist on disk.")
            return

    # 2. Log the annotation as a special category
    # We store the filename in the content so it's searchable
    timestamp = datetime.now().isoformat()
    display_name = os.path.basename(abs_path)
    
    conn.execute(
        "INSERT INTO logs (project_id, timestamp, category, content) VALUES (?, ?, ?, ?)",
        (p_id, timestamp, "ANNOTATION", f"[{display_name}] {note}")
    )
    conn.commit()

    console.print(Panel(
        f"[bold cyan]File:[/bold cyan] {display_name}\n[bold cyan]Note:[/bold cyan] {note}", 
        title="Annotation Saved"
    ))
    
@app.command()
def system():
    """Log system hardware."""
    specs = {"OS": platform.system(), "Cores": os.cpu_count()}
    get_db().execute("INSERT INTO logs (project_id, timestamp, category, content) VALUES (?,?,?,?)", (get_active_project()[0], datetime.now().isoformat(), "SYSTEM", json.dumps(specs))).commit()

@app.command()
def publish():
    """Mock publish command."""
    console.print("[green]Bundle created for sharing.[/green]")

@app.command()
def sweep(size: int = 100):
    """Find large files."""
    archive(dry_run=True) # Re-uses archive logic for finding files

@app.command("generate-man")
def generate_man_page():
    """Generate man page."""
    with open("biln.1", "w") as f: f.write(".TH BILN 1\n.SH NAME\nbiln")
    console.print("Generated biln.1")

# --- MANUAL ---

@app.command()
def manual():
    """Show the comprehensive BILN V1.0 Manual."""
    manual_text = """
    # BILN V1.0 User Manual
    
    **1. Core Management**
    - `init`: Initialize database.
    - `project <name>`: Create/Switch projects.
    - `hello`: Welcome message.
    
    **2. Execution & Logging**
    - `run`: Wrapper for commands. Tracks git, lineage, env.
    - `monitor`: Monitor RAM/CPU of a specific command.
    - `log`: Add a text note.
    - `replay`: Re-run a previous command by ID.
    
    **3. Querying**
    - `history`: Show recent logs.
    - `search <query>`: Find old commands.
    - `lineage <file>`: Trace inputs/outputs.
    - `compare <id1> <id2>`: Diff two runs.
    - `stats`: Project statistics.
    - `show <id>`: Open output file of a run.
    
    **4. Visualization & Impact**
    - `dashboard`: **(New)** Generate HTML report.
    - `viz`: Generate DOT graph of workflow.
    - `samples`: **(New)** Import CSV metadata.
    - `methods`: Auto-write Methods section text.
    - `cite`: List tools for citation.
    
    **5. Reproducibility & Maintenance**
    - `export-snakemake`: Generate Snakemake pipeline.
    - `containerize`: **(New)** Create Dockerfile from history.
    - `archive`: **(New)** Move intermediates to cold storage.
    - `doctor`: **(New)** System sanity check.
    - `verify`: Check file hashes.
    - `snapshot`: Export Conda environment.
    - `export`: Dump data to CSV/JSON.
    - `sweep`: Find large files.
    """
    console.print(Markdown(manual_text))

if __name__ == "__main__":
    app()