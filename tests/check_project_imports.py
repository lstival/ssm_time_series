import os
import sys
import subprocess
from pathlib import Path

def test_all_imports():
    root = Path(__file__).resolve().parents[1]
    src_path = root / "src"
    
    # Add src to sys.path so we can import the package
    sys.path.insert(0, str(src_path))
    
    py_files = list(src_path.rglob("*.py"))
    
    failed = []
    
    print(f"Checking {len(py_files)} files for import/syntax errors...")
    
    for py_file in py_files:
        # Ignore __init__.py if desired, or skip common non-package files
        if "__pycache__" in str(py_file):
            continue
            
        # Get module path relative to src
        rel_path = py_file.relative_to(src_path)
        module_name = ".".join(rel_path.with_suffix("").parts)
        
        # We run in a separate process to avoid polluting this process and to catch NameErrors at top-level
        # Use 'python -c "import module"'
        cmd = [sys.executable, "-c", f"import {module_name}"]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(src_path))
            # print(f"  OK: {module_name}")
        except subprocess.CalledProcessError as e:
            print(f"FAILED: {module_name}")
            print(f"Error: {e.stderr}")
            failed.append((module_name, e.stderr))
            
    if failed:
        print(f"\nTotal failures: {len(failed)}")
        for mod, err in failed:
            print(f"--- {mod} ---")
            print(err)
        sys.exit(1)
    else:
        print("\nAll files imported successfully!")

if __name__ == "__main__":
    test_all_imports()
