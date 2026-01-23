
from huggingface_hub import hf_hub_download, list_repo_files
import os

repo_id = "lstival/CM-Mamba"
subfolder = "cm-mamba-tiny"
filename = "forecasting.py"

print(f"Listing files in {repo_id}...")
try:
    files = list_repo_files(repo_id)
    print("Files found:", files)
    
    if f"{subfolder}/{filename}" in files:
        print(f"File {filename} exists in {subfolder}")
    else:
        print(f"File {filename} NOT found in {subfolder}")

    print(f"Attempting to download {subfolder}/{filename}...")
    path = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder, force_download=True)
    print(f"Downloaded to: {path}")

except Exception as e:
    print(f"Error: {e}")
