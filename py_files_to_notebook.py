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
            "os.makedirs('/content/drive/MyDrive/ILR-IR/data', exist_ok=True)\n",
            "\n",
            "# Clone or download the repository data\n",
            "# Option 1: Using git clone (if git is available)\n",
            "try:\n",
            "    subprocess.run(['git', 'clone', 'https://github.com/pooyaht/ILR-IR.git', '/tmp/ILR-IR'], check=True)\n",
            "    subprocess.run(['cp', '-r', '/tmp/ILR-IR/data/', '/content/drive/MyDrive/ILR-IR/'], shell=True)\n",
            "    print('Data downloaded successfully using git clone')\n",
            "except:\n",
            "    # Option 2: Using wget to download as zip\n",
            "    try:\n",
            "        subprocess.run(['wget', 'https://github.com/pooyaht/ILR-IR/archive/main.zip', '-O', '/tmp/repo.zip'], check=True)\n",
            "        subprocess.run(['unzip', '/tmp/repo.zip', '-d', '/tmp/'], check=True)\n",
            "        subprocess.run(['cp', '-r', '/tmp/ILR-IR-main/data/', '/content/drive/MyDrive/ILR-IR/'], shell=True)\n",
            "        print('Data downloaded successfully using wget')\n",
            "    except Exception as e:\n",
            "        print(f'Error downloading data: {e}')\n",
            "        print('Please manually download the data directory from: https://github.com/pooyaht/ILR-IR/tree/main/data')"
        ]
    }

    notebook["cells"].append(github_download_cell)

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
