"""Exercise the Colab import path in an already-running Python interpreter."""
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("failed_import_already_cached", [False, True])
def test_colab_bootstrap_imports_checkout_without_kernel_restart(tmp_path, failed_import_already_cached):
    content = tmp_path / "content"
    checkout = content / "poodemo"
    shutil.copytree(REPO / "poodemo", checkout / "poodemo", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(REPO / "pyproject.toml", checkout / "pyproject.toml")
    generator = ast.parse((REPO / "scripts" / "build_notebooks.py").read_text())
    bootstrap = next(ast.literal_eval(node.value) for node in generator.body
                     if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "BOOTSTRAP"
                             for target in node.targets))
    bootstrap = bootstrap.replace("/content", str(content))
    # Keep the actual Colab branch and Python imports, replacing only the Drive
    # mount and dependency installation. No network or Google account is used.
    program = f'''
import importlib
import os
from pathlib import Path
import sys
import types

content = Path({str(content)!r})
checkout = content / "poodemo"
sys.path.insert(0, str(content))
if {failed_import_already_cached!r}:
    try:
        import poodemo.pipeline
    except ModuleNotFoundError as error:
        assert error.name == "poodemo.pipeline"
    else:
        raise AssertionError("The fresh interpreter must reproduce the original import failure")
    assert sys.modules["poodemo"].__file__ is None

google = types.ModuleType("google")
google.__path__ = []
colab = types.ModuleType("google.colab")
colab.drive = types.SimpleNamespace(mount=lambda *args, **kwargs: None)
google.colab = colab
sys.modules.update({{"google": google, "google.colab": colab}})
os.environ["POODEMO_SKIP_INSTALL"] = "1"
os.environ["POODEMO_ROOT"] = str(content / "drive" / "run")
namespace = {{}}
exec({bootstrap!r}, namespace)
from poodemo.pipeline import create_run
import poodemo.pipeline
assert Path(poodemo.pipeline.__file__).resolve() == checkout / "poodemo" / "pipeline.py"
run = create_run(namespace["ROOT"], mode="smoke")
assert run.config["schema_version"] == 5

# Re-executing setup after a successful import must keep the loaded package.
loaded = sys.modules["poodemo"]
exec({bootstrap!r}, namespace)
assert sys.modules["poodemo"] is loaded
'''
    completed = subprocess.run([sys.executable, "-I", "-c", program], cwd=content,
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_all_notebooks_share_the_fixed_bootstrap():
    generator = ast.parse((REPO / "scripts" / "build_notebooks.py").read_text())
    bootstrap = next(ast.literal_eval(node.value).strip() for node in generator.body
                     if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "BOOTSTRAP"
                             for target in node.targets))
    for path in (REPO / "notebooks").glob("*.ipynb"):
        notebook = json.loads(path.read_text())
        first_code = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
        assert "".join(first_code["source"]).strip() == bootstrap, path.name
