# BILN: Bioinformatician's Interactive Lab Notebook

**BILN (V4.0)** is a lightweight, "black box" flight recorder for your bioinformatics experiments. It wraps your terminal commands to automatically track **data lineage**, **software versions**, **Git hashes**, and **system resource usage** without getting in your way.

> Stop asking *"Which parameter did I use for that BAM file three months ago?"* Let BILN remember for you.

##  Key Features

*   **Automatic Provenance:** Tracks inputs, outputs, and the exact command used to create them.
*   **Resource Monitoring:** Records Runtime, Peak RAM, and CPU usage for every command.
*   **Reproducibility:** Captures the Git commit hash and tool versions (e.g., `samtools --version`) automatically.
*   **Lineage Tracing:** Query a file to see exactly how it was generated.
*   **Verification:** Checks MD5 hashes to ensure your data hasn't suffered bit-rot.
*   **Reporting:** Exports data to Pandas/JSON or generates Markdown lab reports.

## Installation

### Prerequisites
*   Python 3.8+
*   Git (for version tracking)
*   Conda (recommended)

### Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jimmyXtesla/BILN.git
    cd BILN
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Make it executable:**
    ```bash
    chmod +x BILN.py
    ```

4.  **Add to your path (Optional but Recommended):**
    Add this alias to your `~/.bashrc` or `~/.zshrc` so you can just type `biln`:
    ```bash
    alias biln="BILN.py"
    ```

##  Usage Guide

### 1. Initialize a Project
Start tracking in your current directory. This creates a hidden `.biln` database.
```bash
biln init
biln project my_cancer_study --create
```
2. Run Experiments
Instead of running commands directly, wrap them with biln run.
--inputs: Files read by the command.
--outputs: Files created by the command.

Bash
```
# Example: Mapping reads
biln run --inputs data/sample1.fastq --outputs aligned/sample1.bam \
    "bwa mem ref.fa data/sample1.fastq | samtools sort > aligned/sample1.bam"
```
3. Review History
See what you did today, yesterday, or last month.

Bash
```
biln history

```
4. Trace Data Lineage
Forgot how final_results.vcf was created?

Bash
```
biln lineage final_results.vcf
```
# Output:
```
 > intermediate.bam used in 'samtools mpileup...'
 > raw_reads.fastq used in 'bwa mem...'
```

5. Check System Resources
Run a command and audit how much RAM it consumes.

Bash
```
biln monitor "spades.py -1 R1.fq -2 R2.fq -o assembly_out"
```
Advanced Commands

Command	Description
```
biln verify	Checks if files on disk match their recorded MD5 hashes.
biln snapshot	Exports your current Conda/Pip environment to YAML.
biln compare <ID1> <ID2>	Diff two different runs to see changed parameters.
biln report	Generates a markdown lab notebook summary.
biln manual	Opens the built-in interactive manual.
```

