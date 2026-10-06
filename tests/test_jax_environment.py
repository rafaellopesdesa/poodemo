"""Protect the Colab repair against stale JAX CUDA-plugin discovery.

These checks use real JAX in fresh Python processes, but neither install packages
nor require a GPU. The temporary plugin reproduces the reported registration
failure; the CUDA device environment remains available to independent runtimes
such as PyTorch.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX is optional locally")
def test_colab_bootstrap_removes_discovered_plugin_before_cpu_preflight(tmp_path):
    plugin_root = tmp_path / "plugins"
    plugin = plugin_root / "jax_plugins" / "xla_cuda13"
    plugin.mkdir(parents=True)
    marker = tmp_path / "plugin_initialized"
    (plugin / "__init__.py").write_text(
        "def initialize():\n"
        "    import os\n"
        "    from pathlib import Path\n"
        "    Path(os.environ['POODEMO_PLUGIN_MARKER']).write_text('called')\n"
        "    raise AttributeError(\"module 'jaxlib.xla_client' has no attribute "
        "'register_custom_type_handler'\")\n"
    )
    env = os.environ.copy()
    env.update({
        "JAX_PLATFORMS": "cpu",
        "CUDA_VISIBLE_DEVICES": "0",  # Must stay available to PyTorch.
        "POODEMO_PLUGIN_MARKER": str(marker),
        "PYTHONPATH": str(plugin_root) + os.pathsep + env.get("PYTHONPATH", ""),
    })
    probe = (
        "import json, os, jax, jax.numpy as jnp; "
        "print(json.dumps({'backend': jax.default_backend(), "
        "'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], "
        "'jit_result': float(jax.jit(lambda x: x*x)(jnp.array(3.))) }))"
    )

    def run_probe():
        result = subprocess.run([sys.executable, "-c", probe], env=env,
                                text=True, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stderr
        output = json.loads(result.stdout.strip().splitlines()[-1])
        assert output == {"backend": "cpu", "cuda_visible_devices": "0", "jit_result": 9.0}
        return result

    broken = run_probe()
    assert marker.exists(), "CPU backend selection must not be confused with disabling plugin discovery"
    assert "jax_plugins.xla_cuda13.initialize()" in broken.stderr
    assert "register_custom_type_handler" in broken.stderr

    # Execute the actual Colab bootstrap in a fresh kernel. Mock only Drive and
    # pip: uninstalling the plugin removes its isolated package, while installs
    # are no-ops because this test already runs in the pinned test environment.
    repo = Path(__file__).resolve().parents[1]
    content = tmp_path / "content"
    checkout = content / "poodemo"
    shutil.copytree(repo / "poodemo", checkout / "poodemo",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(repo / "pyproject.toml", checkout / "pyproject.toml")
    generator = ast.parse((repo / "scripts" / "build_notebooks.py").read_text())
    bootstrap = next(ast.literal_eval(node.value) for node in generator.body
                     if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "BOOTSTRAP"
                             for target in node.targets))
    bootstrap = bootstrap.replace("/content", str(content))
    marker.unlink()
    repair_program = f'''
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

calls = []
expected_plugins = {{"jax-cuda12-plugin", "jax-cuda12-pjrt",
                    "jax-cuda13-plugin", "jax-cuda13-pjrt"}}
def fake_pip(command, *args, **kwargs):
    assert command[1:3] == ["-m", "pip"]
    calls.append(command)
    if "uninstall" in command:
        assert "jax" not in sys.modules, "Cleanup must precede the live JAX preflight"
        packages = {{word for word in command[command.index("uninstall")+1:]
                    if not word.startswith("-")}}
        # Removing only these distributions preserves PyTorch and its shared
        # NVIDIA packages. No CUDA driver/package uninstall is permitted here.
        assert packages == expected_plugins
        shutil.rmtree(Path({str(plugin_root)!r}) / "jax_plugins")
    else:
        assert "install" in command
    return 0
subprocess.check_call = fake_pip

google = types.ModuleType("google")
google.__path__ = []
colab = types.ModuleType("google.colab")
colab.drive = types.SimpleNamespace(mount=lambda *args, **kwargs: None)
colab.output = types.SimpleNamespace()  # Imported by the real JAX Colab debugger.
google.colab = colab
sys.modules.update({{"google": google, "google.colab": colab}})
os.environ.pop("POODEMO_SKIP_INSTALL", None)
os.environ["POODEMO_ROOT"] = {str(content / "drive" / "run")!r}
namespace = {{}}
exec({bootstrap!r}, namespace)
assert len(calls) == 3
assert "uninstall" in calls[0]
assert namespace["IN_COLAB"] is True
assert namespace["jax_devices"]
assert all(device.platform == "cpu" for device in namespace["jax_devices"])
assert not Path({str(marker)!r}).exists()
{probe}
'''
    repaired = subprocess.run([sys.executable, "-c", repair_program], env=env,
                              text=True, capture_output=True, timeout=60)
    assert repaired.returncode == 0, repaired.stdout + repaired.stderr
    output = json.loads(repaired.stdout.strip().splitlines()[-1])
    assert output == {"backend": "cpu", "cuda_visible_devices": "0", "jit_result": 9.0}
    assert "Jax plugin configuration error" not in repaired.stderr
