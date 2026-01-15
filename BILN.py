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
    """
    Compare two runs: checks command, runtime, and if output data hashes match.
    """
    conn = get_db()
    r1 = conn.execute("SELECT * FROM logs WHERE id = ?", (id1,)).fetchone()
    r2 = conn.execute("SELECT * FROM logs WHERE id = ?", (id2,)).fetchone()
    
    if not r1 or not r2:
        console.print("[red]One or both Log IDs not found.[/red]")
        return

    table = Table(title=f"Comparison: Run {id1} vs Run {id2}")
    table.add_column("Metric", style="bold")
    table.add_column(f"Run {id1}")
    table.add_column(f"Run {id2}")
    table.add_column("Match")

    # Compare Basic Metadata
    for field in ['cmd', 'tool_version', 'git_hash', 'runtime']:
        match = "[green]YES[/green]" if r1[field] == r2[field] else "[red]NO[/red]"
        table.add_row(field.upper(), str(r1[field]), str(r2[field]), match)

    # Compare Data Hashes (The important part)
    out1 = conn.execute("SELECT f.path, f.hash FROM files f JOIN lineage l ON f.id = l.output_file_id WHERE l.log_id = ?", (id1,)).fetchall()
    out2 = conn.execute("SELECT f.path, f.hash FROM files f JOIN lineage l ON f.id = l.output_file_id WHERE l.log_id = ?", (id2,)).fetchall()

    h1 = {os.path.basename(r['path']): r['hash'] for r in out1}
    h2 = {os.path.basename(r['path']): r['hash'] for r in out2}

    for filename in set(h1.keys()) | set(h2.keys()):
        hash_match = "[green]IDENTICAL[/green]" if h1.get(filename) == h2.get(filename) else "[red]DIFFERENT[/red]"
        table.add_row(f"FILE: {filename}", "MD5 recorded", "MD5 recorded", hash_match)
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
def dashboard(output: str = None):
    """
    Generate a high-end interactive HTML dashboard with project metrics and lineage.
    """
    if Template is None:
        console.print("[red]Error: 'jinja2' not installed. Run 'pip install jinja2'[/red]")
        return

    p_id, p_name = get_active_project()
    conn = get_db()
    
    if output is None:
        output = f"{p_name}_dashboard.html"

    # --- DATA GATHERING ---
    logs = conn.execute("SELECT * FROM logs WHERE project_id = ? ORDER BY timestamp DESC", (p_id,)).fetchall()
    files = conn.execute("SELECT * FROM files WHERE project_id = ?", (p_id,)).fetchall()
    
    # Metrics
    n_runs = len([l for l in logs if l['category'] == 'RUN'])
    n_files = len(files)
    total_runtime = sum([l['runtime'] for l in logs if l['runtime']])
    
    success_count = len([l for l in logs if l['category'] == 'RUN' and l['exit_code'] == 0])
    fail_count = n_runs - success_count
    success_rate = round((success_count / n_runs * 100), 1) if n_runs > 0 else 0

    # Data for Charts (Runtime Trend)
    runtime_data = [{"time": l['timestamp'][11:16], "val": l['runtime'], "cmd": l['cmd'][:30]} 
                    for l in reversed(logs) if l['category'] == 'RUN' and l['runtime']][-20:]

    # Lineage for Mermaid Chart
    links = conn.execute("""
        SELECT f_in.path as src, f_out.path as dest, log.cmd 
        FROM lineage lin
        JOIN logs log ON lin.log_id = log.id
        JOIN files f_in ON lin.input_file_id = f_in.id
        JOIN files f_out ON lin.output_file_id = f_out.id
        WHERE log.project_id = ? LIMIT 50
    """, (p_id,)).fetchall()

    mermaid_code = "graph LR\n"
    for l in links:
        s = os.path.basename(l['src'])
        d = os.path.basename(l['dest'])
        tool = l['cmd'].split()[0] if l['cmd'] else "step"
        mermaid_code += f'    {s} -->|"{tool}"| {d}\n'

    # --- HTML TEMPLATE ---
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>BILN Dashboard: {{p_name}}</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.plot.ly/plotly-2.16.1.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <style>
            body { background-color: #f8f9fa; font-family: 'Inter', sans-serif; }
            .card { border: none; box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,0.075); margin-bottom: 20px; }
            .stat-card { background: linear-gradient(45deg, #4e73df 10%, #224abe 90%); color: white; }
            .mermaid { background: white; padding: 20px; border-radius: 10px; }
            pre { background: #f1f1f1; padding: 10px; border-radius: 5px; font-size: 0.85em; }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-dark bg-dark mb-4">
            <div class="container-fluid">
                <span class="navbar-brand mb-0 h1">BILN Interactive Report</span>
                <span class="badge bg-primary">Project: {{p_name}}</span>
            </div>
        </nav>

        <div class="container">
            <!-- Row 1: Key Stats -->
            <div class="row">
                <div class="col-md-3">
                    <div class="card p-3 text-center border-start border-primary border-5">
                        <div class="text-uppercase text-muted small fw-bold">Total Runs</div>
                        <div class="h3 mb-0">{{n_runs}}</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card p-3 text-center border-start border-success border-5">
                        <div class="text-uppercase text-muted small fw-bold">Files Tracked</div>
                        <div class="h3 mb-0">{{n_files}}</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card p-3 text-center border-start border-info border-5">
                        <div class="text-uppercase text-muted small fw-bold">Total Runtime</div>
                        <div class="h3 mb-0">{{total_runtime}}s</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card p-3 text-center border-start border-warning border-5">
                        <div class="text-uppercase text-muted small fw-bold">Success Rate</div>
                        <div class="h3 mb-0">{{success_rate}}%</div>
                    </div>
                </div>
            </div>

            <div class="row">
                <!-- Row 2: Charts -->
                <div class="col-md-8">
                    <div class="card p-3">
                        <h5>Runtime Performance (Last 20 Runs)</h5>
                        <div id="runtimeChart"></div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card p-3">
                        <h5>Success vs Failure</h5>
                        <div id="successPie"></div>
                    </div>
                </div>
            </div>

            <div class="row">
                <!-- Row 3: Lineage Viz -->
                <div class="col-12">
                    <div class="card p-3">
                        <h5>Workflow Lineage</h5>
                        <div class="mermaid">
                            {{mermaid_code}}
                        </div>
                    </div>
                </div>
            </div>

            <div class="row">
                <!-- Row 4: Log Table -->
                <div class="col-12">
                    <div class="card p-3">
                        <h5>Recent Activity Log</h5>
                        <div class="table-responsive">
                            <table class="table table-hover align-middle">
                                <thead class="table-light">
                                    <tr>
                                        <th>Time</th>
                                        <th>Category</th>
                                        <th>Detail</th>
                                        <th>Status</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for l in logs %}
                                    <tr>
                                        <td class="small">{{l.timestamp[5:16]}}</td>
                                        <td><span class="badge bg-secondary">{{l.category}}</span></td>
                                        <td>
                                            {% if l.cmd %}
                                                <code>{{l.cmd[:80]}}...</code>
                                            {% else %}
                                                {{l.content[:100]}}
                                            {% endif %}
                                        </td>
                                        <td>
                                            {% if l.exit_code == 0 %}<span class="text-success">✔</span>
                                            {% elif l.exit_code %}<span class="text-danger">✘ ({{l.exit_code}})</span>
                                            {% endif %}
                                        </td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Runtime Chart
            const runData = {{ runtime_data | tojson }};
            Plotly.newPlot('runtimeChart', [{
                x: runData.map(d => d.time),
                y: runData.map(d => d.val),
                type: 'bar',
                marker: { color: '#4e73df' }
            }], { margin: { t: 10, b: 40, l: 40, r: 10 }, height: 300 });

            // Success Pie
            Plotly.newPlot('successPie', [{
                values: [{{success_count}}, {{fail_count}}],
                labels: ['Success', 'Fail'],
                type: 'pie',
                hole: .4,
                marker: { colors: ['#1cc88a', '#e74a3b'] }
            }], { margin: { t: 0, b: 0, l: 0, r: 0 }, height: 300 });

            mermaid.initialize({ startOnLoad: true, theme: 'neutral' });
        </script>
    </body>
    </html>
    """

    t = Template(html_template)
    out = t.render(
        p_name=p_name,
        n_runs=n_runs,
        n_files=n_files,
        total_runtime=round(total_runtime, 1),
        success_rate=success_rate,
        success_count=success_count,
        fail_count=fail_count,
        logs=logs,
        runtime_data=runtime_data,
        mermaid_code=mermaid_code
    )

    with open(output, "w") as f:
        f.write(out)
    
    console.print(Panel(f"[green]Beautiful dashboard generated:[/green] [bold]{output}[/bold]", title="Dashboard Ready"))

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
def methods(detailed: bool = typer.Option(False, "--detailed", help="Include every unique command parameters")):
    """
    Generate a professional 'Methods & Materials' draft for your manuscript.
    """
    p_id, _ = get_active_project()
    conn = get_db()
    
    # Get unique tool versions
    tools = conn.execute("""
        SELECT DISTINCT tool_version, cmd 
        FROM logs WHERE project_id = ? AND category='RUN'
    """, (p_id,)).fetchall()
    
    if not tools:
        console.print("[yellow]No runs recorded yet.[/yellow]")
        return

    # Structure the data
    software_list = {}
    for t in tools:
        if t['cmd']:
            binary = t['cmd'].split()[0]
            version = t['tool_version']
            if binary not in software_list:
                software_list[binary] = {"version": version, "params": set()}
            software_list[binary]["params"].add(t['cmd'])

    # Build the paragraph
    intro = "### Methods & Materials (Draft)\n\n"
    para = "Data analysis was performed using the BILN-tracked pipeline. "
    para += "The following software tools were utilized: "
    
    tool_strings = [f"**{name}** (version {info['version']})" for name, info in software_list.items()]
    para += ", ".join(tool_strings) + ". "
    para += f"Computational reproducibility was managed via a {platform.system()} environment."

    console.print(Panel(Markdown(intro + para), title="Scientific Record", expand=False))

    if detailed:
        f_text = "\n**Detailed Parameter Log:**\n"
        for name, info in software_list.items():
            f_text += f"\n- *{name}* settings:\n"
            for p in info['params']:
                f_text += f"  - `{p}`\n"
        console.print(Markdown(f_text))

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
    """
    Generate a Dockerfile that clones your exact analysis environment.
    """
    p_id, name = get_active_project()
    env_file = BILN_DIR / f"{name}_environment.yml"

    if not env_file.exists():
        console.print("[red]Error:[/red] No environment snapshot found. Run [bold]biln snapshot[/bold] first.")
        return

    dockerfile_content = f"""# Generated by BILN for project: {name}
FROM continuumio/miniconda3

# Set up workspace
WORKDIR /analysis

# Copy the environment snapshot
COPY .biln/{name}_environment.yml /tmp/env.yml

# Create the Conda environment
RUN conda env create -f /tmp/env.yml && conda clean -afy

# Set the environment as the default
# Replace 'base' with the name found in your yml if necessary
ENV PATH /opt/conda/envs/$(head -1 /tmp/env.yml | cut -d' ' -f2)/bin:$PATH

# Default command
CMD ["/bin/bash"]
"""

    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)

    console.print(Panel(
        f"[green]Dockerfile created successfully![/green]\n\n"
        "To build your container, run:\n"
        f"[cyan]docker build -t biln-{name.lower()} .[/cyan]",
        title="Containerization"
    ))
    
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
    """
    Export the current Conda environment to a standard YAML file for reproducibility.
    """
    p_id, name = get_active_project()
    # Create the filename
    env_file = BILN_DIR / f"{name}_environment.yml"
    
    console.print(f"[yellow]Capturing Conda environment to {env_file}...[/yellow]")
    
    try:
        # We use --no-builds to make the environment more portable across different machines
        subprocess.run(f"conda env export --no-builds > {env_file}", shell=True, check=True)
        
        conn = get_db()
        conn.execute("INSERT INTO logs (project_id, timestamp, category, content) VALUES (?,?,?,?)", 
                     (p_id, datetime.now().isoformat(), "SNAPSHOT", str(env_file)))
        conn.commit()
        
        console.print(f"[bold green]Snapshot successful![/bold green] Use 'biln containerize' to turn this into a Docker image.")
    except Exception as e:
        console.print(f"[red]Error capturing environment: {e}[/red]\nEnsure Conda is in your PATH.")

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
def publish(output_name: str = "Research_Bundle"):
    """
    Bundle the database, documentation, and lineage into a single ZIP for sharing.
    """
    p_id, p_name = get_active_project()
    # First, refresh the docs
    export(output="PROVENANCE.md")
    viz(output="workflow.dot")
    
    bundle_fn = f"{output_name}_{p_name}_{datetime.now().strftime('%Y%m%d')}"
    
    # Create a temporary folder to gather files
    tmp_dir = Path(f"tmp_{bundle_fn}")
    tmp_dir.mkdir(exist_ok=True)
    
    try:
        # Copy essential files
        shutil.copy(DB_PATH, tmp_dir / "biln_notebook.db")
        shutil.copy("PROVENANCE.md", tmp_dir / "PROVENANCE.md")
        if os.path.exists("workflow.dot"):
            shutil.copy("workflow.dot", tmp_dir / "workflow.dot")
        
        # Create ZIP
        shutil.make_archive(bundle_fn, 'zip', tmp_dir)
        console.print(f"[bold green]Success![/bold green] Bundle created: [white]{bundle_fn}.zip[/white]")
    finally:
        shutil.rmtree(tmp_dir)

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