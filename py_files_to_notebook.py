#!/usr/bin/env python3
"""
Script to combine all Python files in a model directory into a single Jupyter notebook.
Each Python file becomes a separate cell in the notebook.
"""

import os
import json
import glob
from pathlib import Path


def create_notebook_from_python_files(model_directory, output_notebook_path):
    """
    Create a Jupyter notebook from all Python files in the specified directory.

    Args:
        model_directory (str): Path to the directory containing Python files
        output_notebook_path (str): Path where the notebook will be saved
    """

    # Initialize notebook structure
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    # First cell: Mount Google Drive
    mount_drive_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Mount Google Drive\n",
            "from google.colab import drive\n",
            "drive.mount('/content/drive')\n",
            "print('Google Drive mounted successfully!')"
        ]
    }

    notebook["cells"].append(mount_drive_cell)

    # Second cell: GitHub data directory download command
    github_download_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Download data directory from GitHub repository\n",
            "import os\n",
            "import subprocess\n",
            "\n",
            "# Create target directory\n",
            "os.makedirs('/content/drive/MyDrive/ILR-IR', exist_ok=True)\n",
            "\n",
            "# Clone repository and copy data\n",
            "try:\n",
            "    subprocess.run(['git', 'clone', 'https://github.com/pooyaht/ILR-IR.git', '/tmp/ILR-IR'], check=True)\n",
            "    print('Repository cloned successfully')\n",
            "    \n",
            "    # Copy data using shell command\n",
            "    !cp -vr /tmp/ILR-IR/data /content/drive/MyDrive/ILR-IR/\n",
            "    \n",
            "    # Change to the project directory\n",
            "    os.chdir('/content/drive/MyDrive/ILR-IR/')\n",
            "    print(f'Changed working directory to: {os.getcwd()}')\n",
            "    print('Data downloaded and setup completed successfully!')\n",
            "    \n",
            "except:\n",
            "    print('Git clone failed. Please manually download the data from: https://github.com/pooyaht/ILR-IR/tree/main/data')"
        ]
    }

    notebook["cells"].append(github_download_cell)

    # Third cell: Install pyHGT and dependencies
    install_pyhgt_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Install pyHGT and required dependencies\n",
            "!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n",
            "!pip install pyHGT\n",
            "!pip install networkx\n",
            "!pip install scikit-learn\n",
            "!pip install more-itertools\n",
            "!pip install dill\n",
            "!pip install pandas\n",
            "!pip install numpy\n",
            "!pip install tqdm\n",
            "!pip install seaborn\n",
            "!pip install matplotlib\n",
            "print('All dependencies installed successfully!')"
        ]
    }

    notebook["cells"].append(install_pyhgt_cell)

    # Fourth cell: Verify installation and imports
    verify_imports_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Verify pyHGT installation and import required modules\n",
            "try:\n",
            "    import torch\n",
            "    import torch.nn as nn\n",
            "    import torch.nn.functional as F\n",
            "    import numpy as np\n",
            "    import networkx as nx\n",
            "    import pandas as pd\n",
            "    from sklearn.utils import shuffle\n",
            "    from more_itertools import flatten\n",
            "    import dill\n",
            "    from tqdm import tqdm\n",
            "    import seaborn as sb\n",
            "    import matplotlib.pyplot as plt\n",
            "    from pyHGT.model import *\n",
            "    from pyHGT.data import *\n",
            "    print('✅ All imports successful!')\n",
            "    print(f'PyTorch version: {torch.__version__}')\n",
            "    print(f'CUDA available: {torch.cuda.is_available()}')\n",
            "    if torch.cuda.is_available():\n",
            "        print(f'CUDA device: {torch.cuda.get_device_name(0)}')\n",
            "except ImportError as e:\n",
            "    print(f'❌ Import error: {e}')\n",
            "    print('Please check if all dependencies are installed correctly.')"
        ]
    }

    notebook["cells"].append(verify_imports_cell)

    # Get current script name to exclude it
    current_script = os.path.abspath(__file__)

    # Get all Python files recursively, excluding __init__.py and this script
    python_files = []
    for root, dirs, files in os.walk(model_directory):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                full_path = os.path.join(root, file)
                # Exclude this script itself
                if os.path.abspath(full_path) != current_script:
                    python_files.append(full_path)

    python_files.sort()  # Sort files alphabetically

    if not python_files:
        print(f"No Python files found in directory: {model_directory}")
        return

    print(f"Found {len(python_files)} Python files:")
    for py_file in python_files:
        rel_path = os.path.relpath(py_file, model_directory)
        print(f"  - {rel_path}")

    # Process each Python file
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                file_content = f.read()

            # Create a cell for this Python file
            rel_path = os.path.relpath(py_file, model_directory)

            # Split content into lines and ensure each line ends with \n except the last one
            content_lines = file_content.split('\n')
            source_lines = [
                f"# File: {rel_path}\n",
                f"# Source: {py_file}\n",
                "\n"
            ]

            # Add each line with proper line endings
            for i, line in enumerate(content_lines):
                if i == len(content_lines) - 1 and line == "":
                    # Skip empty last line
                    continue
                elif i == len(content_lines) - 1:
                    # Last line without \n
                    source_lines.append(line)
                else:
                    # All other lines with \n
                    source_lines.append(line + "\n")

            # Fix argparse for notebook compatibility by replacing parse_args() calls
            fixed_content = []
            in_main_block = False

            for line_with_ending in source_lines:
                line = line_with_ending.rstrip('\n')

                # Check if we're in the main block
                if "if __name__ == '__main__':" in line:
                    in_main_block = True

                # Replace problematic argparse lines
                if "args = args.parse_args()" in line:
                    # Replace with notebook-compatible version
                    fixed_content.append(
                        "    # Notebook-compatible argument parsing\n")
                    fixed_content.append("    import sys\n")
                    fixed_content.append(
                        "    if 'ipykernel' in sys.modules:\n")
                    fixed_content.append(
                        "        # Running in notebook - use defaults\n")
                    fixed_content.append(
                        "        args = argparse.Namespace(\n")
                    fixed_content.append(
                        "            data='./data/ICEWS14_forecasting',\n")
                    fixed_content.append("            state='train',\n")
                    fixed_content.append("            ratio=1,\n")
                    fixed_content.append("            his_len=50\n")
                    fixed_content.append("        )\n")
                    fixed_content.append("    else:\n")
                    fixed_content.append(
                        "        # Running as script - use command line args\n")
                    fixed_content.append("        args = args.parse_args()\n")
                elif "args = parse_args()" in line and not in_main_block:
                    # This is the global args = parse_args() call
                    fixed_content.append(
                        "# Notebook-compatible argument setup\n")
                    fixed_content.append("import sys\n")
                    fixed_content.append("if 'ipykernel' in sys.modules:\n")
                    fixed_content.append(
                        "    # Running in notebook - create args with defaults\n")
                    fixed_content.append("    import argparse\n")
                    fixed_content.append("    args = argparse.Namespace(\n")
                    fixed_content.append(
                        "        data='./data/ICEWS14_forecasting',\n")
                    fixed_content.append("        state='train',\n")
                    fixed_content.append("        ratio=1,\n")
                    fixed_content.append("        his_len=50\n")
                    fixed_content.append("    )\n")
                    fixed_content.append("else:\n")
                    fixed_content.append(
                        "    # Running as script - use command line args\n")
                    fixed_content.append("    args = parse_args()\n")
                else:
                    # Keep the original line
                    if line_with_ending.endswith('\n'):
                        fixed_content.append(line_with_ending)
                    else:
                        fixed_content.append(line_with_ending)

            source_lines = fixed_content

            cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source_lines
            }

            notebook["cells"].append(cell)
            print(f"Added {rel_path} to notebook")

        except Exception as e:
            print(f"Error processing {py_file}: {e}")

    # Save the notebook
    try:
        with open(output_notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        print(f"\nNotebook saved successfully to: {output_notebook_path}")
        print(f"Total cells: {len(notebook['cells'])}")

    except Exception as e:
        print(f"Error saving notebook: {e}")


def main():
    """Main function to run the script."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert Python files to Jupyter notebook')
    parser.add_argument('model_directory',
                        help='Directory containing Python files')
    parser.add_argument('-o', '--output', default='combined_model.ipynb',
                        help='Output notebook filename (default: combined_model.ipynb)')

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.model_directory):
        print(f"Error: Directory '{args.model_directory}' does not exist")
        return

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(
        args.output) if os.path.dirname(args.output) else '.'
    os.makedirs(output_dir, exist_ok=True)

    # Create the notebook
    create_notebook_from_python_files(args.model_directory, args.output)


if __name__ == "__main__":
    main()
