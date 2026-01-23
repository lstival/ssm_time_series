import os
import re
from pathlib import Path

package_name = "ssm_time_series"
root_dir = Path(__file__).resolve().parent

def fix_file(file_path):
    if file_path == __file__:
        return
        
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    new_content = content
    
    # 1. Broad replacements for moved packages (already done mostly, but ensuring consistency)
    replacements = {
        r"import ssm_time_series\.utils\.general as u": f"from {package_name} import utils as u",
        r"import ssm_time_series\.training\.utils as tu": f"from {package_name} import training as tu",
        r"from ssm_time_series.data.utils import": f"from {package_name}.data.utils import", # Some shared logic moved there
    }
    
    for pattern, repl in replacements.items():
        new_content = re.sub(pattern, repl, new_content)

    if new_content != content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Fixed {file_path}")

for root, dirs, files in os.walk(root_dir):
    if any(p in root for p in [".git", "__pycache__", "build", "dist", "ssm_time_series.egg-info"]):
        continue
    for file in files:
        if file.endswith(".py"):
            fix_file(os.path.join(root, file))
