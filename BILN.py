# BILN - Python-based Interactive Lab Notebook for Bioinformaticians
#Author: Jimmy X.

import os
import sqlite3
import json
from datetime import datetime
import pandas as pd
import numpy as np
from Bio import SeqIO
# from Bio.SeqUtils import GC
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import re


class Experiment:
    """Manages a single bioinformatics experiment."""

    def __init__(self, name, description, dataset_metadata):
        """
        Initializes a new Experiment.

        Args:
            name (str): The name of the experiment.
            description (str): A brief description of the experiment.
            dataset_metadata (dict): Metadata about the dataset being used.
        """
        self.id = None
        self.name = name
        self.description = description
        self.dataset_metadata = json.dumps(dataset_metadata)
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.steps = []

    def add_step(self, step):
        """
        Adds a new analysis step to the experiment.

        Args:
            step (Step): The Step object to add.
        """
        self.steps.append(step)

    def save(self, conn):
        """
        Saves the experiment to the database.

        Args:
            conn: SQLite database connection object.
        """
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO experiments (name, description, dataset_metadata, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (self.name, self.description, self.dataset_metadata, self.timestamp))
        self.id = cursor.lastrowid
        conn.commit()
        print(f"Experiment '{self.name}' saved with ID: {self.id}")

    @staticmethod
    def view_all(conn):
        """
        Retrieves and displays all experiments from the database.

        Args:
            conn: SQLite database connection object.
        """
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM experiments")
        rows = cursor.fetchall()
        if not rows:
            print("No experiments found.")
            return
        print("\n--- All Experiments ---")
        for row in rows:
            print(f"ID: {row[0]}, Name: {row[1]}, Description: {row[2]}, Timestamp: {row[4]}")
        print("-----------------------\n")

    @staticmethod
    def load(conn, experiment_id):
        """
        Loads an experiment and its steps from the database.

        Args:
            conn: SQLite database connection object.
            experiment_id (int): The ID of the experiment to load.

        Returns:
            Experiment: The loaded Experiment object.
        """
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
        row = cursor.fetchone()
        if row:
            exp = Experiment(row[1], row[2], json.loads(row[3]))
            exp.id = row[0]
            exp.timestamp = row[4]

            cursor.execute("SELECT * FROM steps WHERE experiment_id = ?", (experiment_id,))
            steps_data = cursor.fetchall()
            for step_row in steps_data:
                step = Step(step_row[2], json.loads(step_row[3]), json.loads(step_row[4]))
                step.id = step_row[0]
                step.timestamp = step_row[5]
                exp.add_step(step)
            return exp
        return None


class Step:
    """Represents a single analysis step in an experiment."""

    def __init__(self, action_name, input_summary, output_summary):
        """
        Initializes a new Step.

        Args:
            action_name (str): The name of the analysis action performed.
            input_summary (dict): A summary of the input data.
            output_summary (dict): A summary of the output results.
        """
        self.id = None
        self.action_name = action_name
        self.input_summary = json.dumps(input_summary)
        self.output_summary = json.dumps(output_summary)
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def save(self, conn, experiment_id):
        """
        Saves the step to the database.

        Args:
            conn: SQLite database connection object.
            experiment_id (int): The ID of the parent experiment.
        """
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO steps (experiment_id, action_name, input_summary, output_summary, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (experiment_id, self.action_name, self.input_summary, self.output_summary, self.timestamp))
        self.id = cursor.lastrowid
        conn.commit()


class Analysis:
    """Contains methods for various bioinformatics analyses."""

    def __init__(self, experiment):
        self.experiment = experiment
        self.project_folder = f"experiment_{self.experiment.id}_{self.experiment.name.replace(' ', '_')}"
        if not os.path.exists(self.project_folder):
            os.makedirs(self.project_folder)

    def _log_step(self, conn, action_name, input_summary, output_summary):
        """Creates and saves a Step object."""
        step = Step(action_name, input_summary, output_summary)
        step.save(conn, self.experiment.id)
        self.experiment.add_step(step)
        print(f"Logged step: {action_name}")

    def get_sequence_stats(self, conn, file_path):
        """
        Calculates basic statistics for a sequence file.

        Args:
            conn: SQLite database connection object.
            file_path (str): The path to the FASTA or FASTQ file.
        """
        try:
            file_type = "fasta" if file_path.endswith((".fasta", ".fa")) else "fastq"
            records = list(SeqIO.parse(file_path, file_type))
            if not records:
                print("Error: No sequences found in the file.")
                return

            lengths = [len(rec) for rec in records]
            #gc_contents = [GC(rec.seq) for rec in records]

            stats = {
                "total_sequences": len(records),
                "avg_length": np.mean(lengths),
                "max_length": np.max(lengths),
                "min_length": np.min(lengths)
                
            }

            # Visualization
            self._plot_length_distribution(lengths)

            input_summary = {"file_path": file_path}
            output_summary = {"stats": stats, "plots": [os.path.join(self.project_folder, "length_distribution.png"),
                                                        os.path.join(self.project_folder, "gc_content.png")]}
            self._log_step(conn, "Sequence Stats", input_summary, output_summary)

        except Exception as e:
            print(f"An error occurred: {e}")

    def motif_search(self, conn, file_path, motif):
        """
        Searches for a given motif in a FASTA or CSV file.

        Args:
            conn: SQLite database connection object.
            file_path (str): The path to the input file.
            motif (str): The motif to search for.
        """
        found_motifs = []
        if file_path.endswith((".fasta", ".fa", ".fastq")):
            file_type = "fasta" if file_path.endswith((".fasta", ".fa")) else "fastq"
            for record in SeqIO.parse(file_path, file_type):
                for match in re.finditer(motif, str(record.seq)):
                    found_motifs.append({
                        "sequence_id": record.id,
                        "start": match.start(),
                        "end": match.end()
                    })
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            # Assuming the sequence is in a column named 'sequence'
            for index, row in df.iterrows():
                for match in re.finditer(motif, row['sequence']):
                    found_motifs.append({
                        "row_index": index,
                        "start": match.start(),
                        "end": match.end()
                    })
        else:
            print("Unsupported file type for motif search.")
            return

        input_summary = {"file_path": file_path, "motif": motif}
        output_summary = {"found_motifs": found_motifs}
        self._log_step(conn, "Motif Search", input_summary, output_summary)

    def rna_seq_qc(self, conn, counts_file):
        """
        Generates a basic QC summary for RNA-seq count data.

        Args:
            conn: SQLite database connection object.
            counts_file (str): Path to a CSV file with gene counts.
        """
        try:
            counts_df = pd.read_csv(counts_file, index_col=0)
            summary = {
                "num_genes": counts_df.shape[0],
                "num_samples": counts_df.shape[1],
                "total_counts_per_sample": counts_df.sum().to_dict()
            }

            # Visualization
            self._plot_expression_heatmap(counts_df)

            input_summary = {"counts_file": counts_file}
            output_summary = {"summary": summary, "heatmap": os.path.join(self.project_folder, "expression_heatmap.png")}
            self._log_step(conn, "RNA-seq QC", input_summary, output_summary)

        except Exception as e:
            print(f"An error occurred during RNA-seq QC: {e}")

    def _plot_length_distribution(self, lengths):
        """Generates and saves a plot of sequence length distribution."""
        plt.figure(figsize=(10, 6))
        sns.histplot(lengths, kde=True)
        plt.title("Sequence Length Distribution")
        plt.xlabel("Sequence Length (bp)")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(self.project_folder, "length_distribution.png"))
        plt.close()

    def _plot_gc_content(self, gc_contents):
        """Generates and saves a plot of GC content."""
        plt.figure(figsize=(10, 6))
        sns.histplot(gc_contents, kde=True)
        plt.title("GC Content Distribution")
        plt.xlabel("GC Content (%)")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(self.project_folder, "gc_content.png"))
        plt.close()

    def _plot_expression_heatmap(self, counts_df):
        """Generates and saves an expression heatmap."""
        plt.figure(figsize=(12, 10))
        sns.heatmap(np.log1p(counts_df.head(50)), cmap="viridis") # Heatmap of top 50 genes
        plt.title("Gene Expression Heatmap (Top 50 Genes)")
        plt.xlabel("Samples")
        plt.ylabel("Genes")
        plt.savefig(os.path.join(self.project_folder, "expression_heatmap.png"))
        plt.close()

# --- Database & Persistence ---

def setup_database():
    """Sets up the SQLite database and creates tables."""
    conn = sqlite3.connect("biln.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            dataset_metadata TEXT,
            timestamp TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER,
            action_name TEXT NOT NULL,
            input_summary TEXT,
            output_summary TEXT,
            timestamp TEXT,
            FOREIGN KEY (experiment_id) REFERENCES experiments (id)
        )
    ''')
    conn.commit()
    return conn

# --- Reporting ---

def export_report(experiment, format='md'):
    """
    Exports the experiment notebook to a file.

    Args:
        experiment (Experiment): The experiment to report on.
        format (str): The desired output format ('md', 'html', 'pdf').
    """
    project_folder = f"experiment_{experiment.id}_{experiment.name.replace(' ', '_')}"
    report_path = os.path.join(project_folder, f"report.{format}")

    if format == 'md':
        with open(report_path, 'w') as f:
            f.write(f"# Experiment Report: {experiment.name}\n\n")
            f.write(f"**ID:** {experiment.id}\n")
            f.write(f"**Description:** {experiment.description}\n")
            f.write(f"**Date:** {experiment.timestamp}\n")
            f.write(f"**Dataset Metadata:** `{experiment.dataset_metadata}`\n\n")
            f.write("---\n\n")

            for i, step in enumerate(experiment.steps):
                f.write(f"## Step {i+1}: {step.action_name}\n\n")
                f.write(f"**Timestamp:** {step.timestamp}\n\n")
                f.write("### Input\n")
                f.write(f"```json\n{json.dumps(json.loads(step.input_summary), indent=2)}\n```\n\n")
                f.write("### Output\n")
                output_data = json.loads(step.output_summary)
                # Embed plots in markdown
                if 'plots' in output_data:
                    for plot_path in output_data['plots']:
                        f.write(f"![Plot]({os.path.basename(plot_path)})\n\n")
                if 'heatmap' in output_data:
                     f.write(f"![Heatmap]({os.path.basename(output_data['heatmap'])})\n\n")

                f.write(f"```json\n{json.dumps(output_data, indent=2)}\n```\n\n")
                f.write("---\n\n")
    
    elif format == 'pdf':
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Experiment Report: {experiment.name}", ln=True, align='C')
        
        pdf.cell(200, 10, txt=f"ID: {experiment.id}", ln=True)
        pdf.cell(200, 10, txt=f"Description: {experiment.description}", ln=True)
        pdf.cell(200, 10, txt=f"Date: {experiment.timestamp}", ln=True)
        
        for i, step in enumerate(experiment.steps):
            pdf.add_page()
            pdf.cell(200, 10, txt=f"Step {i+1}: {step.action_name}", ln=True)
            output_data = json.loads(step.output_summary)
            if 'plots' in output_data:
                for plot_path in output_data['plots']:
                    if os.path.exists(plot_path):
                        pdf.image(plot_path, w=150)
            if 'heatmap' in output_data and os.path.exists(output_data['heatmap']):
                 pdf.image(output_data['heatmap'], w=150)
        
        pdf.output(report_path)

    print(f"Report exported to {report_path}")

# --- Console UI ---

def main_menu():
    print("\nWelcome to the Python-based Interactive Lab Notebook for Bioinformaticians (BILN)")
    print("1. Start a new experiment")
    print("2. View and continue a previous experiment")
    print("3. Exit")
    return input("Choose an option: ")

def experiment_menu(analysis):
    while True:
        print("\n--- Experiment Menu ---")
        print("1. Get sequence stats")
        print("2. Search for a motif")
        print("3. Perform RNA-seq QC")
        print("4. Export report")
        print("5. Back to main menu")
        choice = input("Choose an action: ")

        if choice == '1':
            file_path = input("Enter the path to your FASTA/FASTQ file: ")
            if os.path.exists(file_path):
                analysis.get_sequence_stats(conn, file_path)
            else:
                print("File not found.")
        elif choice == '2':
            file_path = input("Enter the path to your FASTA/CSV file: ")
            motif = input("Enter the motif to search for: ")
            if os.path.exists(file_path):
                analysis.motif_search(conn, file_path, motif)
            else:
                print("File not found.")
        elif choice == '3':
            counts_file = input("Enter the path to your RNA-seq counts CSV file: ")
            if os.path.exists(counts_file):
                analysis.rna_seq_qc(conn, counts_file)
            else:
                print("File not found.")
        elif choice == '4':
            fmt = input("Enter report format (md/pdf): ").lower()
            if fmt in ['md', 'pdf']:
                export_report(analysis.experiment, fmt)
            else:
                print("Invalid format.")
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please try again.")

# --- Example Usage ---

if __name__ == "__main__":
    # Setup
    conn = setup_database()

    # Create dummy files for demonstration
    if not os.path.exists("example.fasta"):
        with open("example.fasta", "w") as f:
            f.write(">seq1\n")
            f.write("ATGCGCATGCATGCATGC\n")
            f.write(">seq2\n")
            f.write("CGATGCATGCTAGCTACG\n")

    if not os.path.exists("counts.csv"):
        counts_data = {'gene': ['gene1', 'gene2', 'gene3'],
                       'sample1': [10, 20, 5],
                       'sample2': [15, 25, 8]}
        pd.DataFrame(counts_data).set_index('gene').to_csv("counts.csv")

    while True:
        choice = main_menu()
        if choice == '1':
            exp_name = input("Enter experiment name: ")
            exp_desc = input("Enter experiment description: ")
            dataset_meta = {"file": "example.fasta", "source": "in-silico"}
            new_exp = Experiment(exp_name, exp_desc, dataset_meta)
            new_exp.save(conn)
            analysis_tool = Analysis(new_exp)
            experiment_menu(analysis_tool)
        elif choice == '2':
            Experiment.view_all(conn)
            exp_id = input("Enter the ID of the experiment to continue: ")
            loaded_exp = Experiment.load(conn, int(exp_id))
            if loaded_exp:
                print(f"Continuing experiment: {loaded_exp.name}")
                analysis_tool = Analysis(loaded_exp)
                experiment_menu(analysis_tool)
            else:
                print("Experiment not found.")
        elif choice == '3':
            break
        else:
            print("Invalid choice.")

    conn.close()

