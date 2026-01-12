"""Unit tests to verify project structure and import integrity."""

import subprocess
import sys
from pathlib import Path
import pytest

# Root of the repository
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

def get_python_files():
    """Discover all python files in the src directory."""
    return [
        p.relative_to(SRC_DIR) 
        for p in SRC_DIR.rglob("*.py") 
        if "__pycache__" not in str(p) and not str(p.name).startswith(".")
    ]

@pytest.mark.parametrize("rel_path", get_python_files())
def test_import_stability(rel_path):
    """
    Test that each file in the src directory can be imported without errors.
    This catches NameErrors (missing imports), ModuleNotFoundErrors, and SyntaxErrors.
    """
    module_name = ".".join(rel_path.with_suffix("").parts)
    
    # Run import in a subprocess to ensure isolation
    cmd = [
        sys.executable, 
        "-c", 
        f"import sys; sys.path.insert(0, r'{SRC_DIR}'); import {module_name}"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    assert result.returncode == 0, (
        f"Failed to import {module_name} (from {rel_path})\n"
        f"STDOUT: {result.stdout}\n"
        f"STDERR: {result.stderr}"
    )

def test_package_installable():
    """Verify the package can be detected if src is in path."""
    sys.path.insert(0, str(SRC_DIR))
    try:
        import ssm_time_series
        assert ssm_time_series.__version__ is not None
    except ImportError:
        pytest.fail("ssm_time_series package not found in src/")
    except AttributeError:
        # __version__ might not be defined in __init__.py yet, but import should work
        pass
