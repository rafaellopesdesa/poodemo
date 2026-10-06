"""Exercise the actual Colab setup without installing packages or needing a GPU.

CPU tests run real JAX and reproduce the reported stale CUDA13 plugin failure.
GPU tests explicitly simulate device availability/JAX so they check setup routing
and fail-fast behavior, not CUDA execution or GPU performance.
"""
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
PLUGIN_NAMES = ("jax-cuda12-plugin", "jax-cuda12-pjrt", "jax-cuda13-plugin", "jax-cuda13-pjrt")
GPU_PINS = {"jax==0.5.3", "jaxlib==0.5.3",
            "jax-cuda12-plugin[with-cuda]==0.5.3", "jax-cuda12-pjrt==0.5.3"}
COLAB_MOCK = '''
google = types.ModuleType("google")
google.__path__ = []
colab = types.ModuleType("google.colab")
colab.drive = types.SimpleNamespace(mount=lambda *args, **kwargs: None)
colab.output = types.SimpleNamespace()  # Imported by the real JAX Colab debugger.
google.colab = colab
sys.modules.update({"google": google, "google.colab": colab})
'''


def bootstrap_source(tmp_path):
    content = tmp_path / "content"
    checkout = content / "poodemo"
    shutil.copytree(REPO / "poodemo", checkout / "poodemo",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(REPO / "pyproject.toml", checkout / "pyproject.toml")
    source = ast.parse((REPO / "scripts" / "build_notebooks.py").read_text())
    bootstrap = next(ast.literal_eval(node.value) for node in source.body
                     if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "BOOTSTRAP" for t in node.targets))
    return bootstrap.replace("/content", str(content))


def subprocess_env(tmp_path, backend):
    env = os.environ.copy()
    env.pop("POODEMO_SKIP_INSTALL", None)
    env.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
    env.update({"POODEMO_JAX_BACKEND": backend,
                "POODEMO_ROOT": str(tmp_path / "run"), "CUDA_VISIBLE_DEVICES": "0"})
    return env


def execute(program, env):
    result = subprocess.run([sys.executable, "-c", program], env=env,
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX is optional locally")
@pytest.mark.parametrize("backend,gpu_present", [("auto", False), ("cpu", True)])
def test_real_cpu_cleanup_precedes_preflight(tmp_path, backend, gpu_present):
    """CPU selection alone still discovers plugins; actual setup removes them."""
    bootstrap = bootstrap_source(tmp_path)
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
    # Real distribution metadata drives the actual bootstrap's cleanup decision.
    for name in PLUGIN_NAMES:
        version = "0.5.3" if "cuda12" in name else "0.8.2"
        metadata = plugin_root / f"{name.replace('-', '_')}-{version}.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    env = subprocess_env(tmp_path, backend)
    env.update({"JAX_PLATFORMS": "cpu", "POODEMO_PLUGIN_MARKER": str(marker),
                "PYTHONPATH": str(plugin_root) + os.pathsep + env.get("PYTHONPATH", "")})
    broken = execute("import jax; assert jax.default_backend() == 'cpu'", env)
    assert marker.exists()
    assert "jax_plugins.xla_cuda13.initialize()" in broken.stderr
    assert "register_custom_type_handler" in broken.stderr
    marker.unlink()

    program = f'''
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types
{COLAB_MOCK}
calls = []
def fake_pip(command, *args, **kwargs):
    assert command[1:3] == ["-m", "pip"]
    calls.append(command)
    if "uninstall" in command:
        assert "jax" not in sys.modules, "Cleanup must precede real JAX initialization"
        packages = {{s for s in command[command.index("uninstall")+1:] if not s.startswith("-")}}
        assert packages == {set(PLUGIN_NAMES)!r}
        shutil.rmtree(Path({str(plugin_root)!r}) / "jax_plugins")
        for path in Path({str(plugin_root)!r}).glob("*.dist-info"):
            shutil.rmtree(path)
    else:
        assert "install" in command
        assert not any("jax-cuda" in value for value in command)
    return 0
subprocess.check_call = fake_pip
subprocess.run = lambda *a, **kw: types.SimpleNamespace(
    returncode=0 if {gpu_present!r} else 1,
    stdout="Simulated NVIDIA GPU" if {gpu_present!r} else "")
namespace = {{}}
exec({bootstrap!r}, namespace)
assert namespace["selected_backend"] == "cpu"
assert namespace["jax_devices"]
assert all(device.platform == "cpu" for device in namespace["jax_devices"])
assert namespace["probe_gradient"].dtype.name == "float64"
assert list(namespace["probe_gradient"]) == [2., 4., 6.]
assert len(calls) == 3 and "uninstall" in calls[0]
assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
assert not Path({str(marker)!r}).exists()
'''
    repaired = execute(program, env)
    assert "Jax plugin configuration error" not in repaired.stderr
    assert "float64 JIT/gradient OK" in repaired.stdout


@pytest.mark.parametrize("backend,installed12,device_platform,should_fail", [
    ("auto", "0.5.3", "gpu", False),
    ("gpu", "0.4.38", "gpu", False),
    ("gpu", "0.5.3", "cpu", True),
])
def test_simulated_gpu_routing_pins_cleanup_and_no_silent_fallback(
        tmp_path, backend, installed12, device_platform, should_fail):
    """Simulate hardware/JAX to test routing; this is not a CUDA execution test."""
    bootstrap = bootstrap_source(tmp_path)
    program = f'''
import importlib.metadata as metadata
import os
import subprocess
import sys
import types
import numpy as np
{COLAB_MOCK}
installed = {{name: ({installed12!r} if "cuda12" in name else "0.8.2")
             for name in {PLUGIN_NAMES!r}}}
real_version = metadata.version
metadata.version = lambda name: installed[name] if name in installed else real_version(name)
events, calls = [], []
def fake_pip(command, *args, **kwargs):
    assert command[1:3] == ["-m", "pip"]
    calls.append(command)
    assert "preflight" not in events, "Resolve packages before probing JAX"
    if "uninstall" in command:
        removed = {{s for s in command[command.index("uninstall")+1:] if not s.startswith("-")}}
        expected = {{"jax-cuda13-plugin", "jax-cuda13-pjrt"}}
        if {installed12!r} != "0.5.3":
            expected.update(["jax-cuda12-plugin", "jax-cuda12-pjrt"])
        assert removed == expected
        for name in removed:
            installed.pop(name)
        events.append("uninstall")
    else:
        assert "install" in command and "--upgrade" not in command
        if "-r" in command:
            assert {GPU_PINS!r} <= set(command)
            installed.update({{"jax-cuda12-plugin": "0.5.3", "jax-cuda12-pjrt": "0.5.3"}})
        events.append("install")
    return 0
subprocess.check_call = fake_pip
subprocess.run = lambda *a, **kw: types.SimpleNamespace(returncode=0, stdout="Simulated NVIDIA GPU")

# Implement only the JAX surface exercised by the real bootstrap preflight.
# Device behavior is deliberately simulated; mathematical execution uses NumPy.
configuration = {{}}
class Gradient(np.ndarray):
    def block_until_ready(self):
        return self
    def devices(self):
        return [types.SimpleNamespace(platform={device_platform!r})]
def devices():
    events.append("preflight")
    return [types.SimpleNamespace(platform={device_platform!r})]
jax = types.ModuleType("jax")
jax.__version__ = "0.5.3"
jax.config = types.SimpleNamespace(update=lambda name, value: configuration.update({{name: value}}))
jax.devices = devices
jax.jit = lambda fun: fun
jax.value_and_grad = lambda fun: lambda x: (fun(x), (2*x).view(Gradient))
jax.numpy = np
jaxlib = types.ModuleType("jaxlib")
jaxlib.__version__ = "0.5.3"
sys.modules.update({{"jax": jax, "jax.numpy": np, "jaxlib": jaxlib}})
namespace = {{}}
error = None
try:
    exec({bootstrap!r}, namespace)
except RuntimeError as caught:
    error = caught
assert (error is not None) == {should_fail!r}
if error is not None:
    assert "JAX gpu startup failed" in str(error)
assert namespace["selected_backend"] == "gpu"
assert os.environ["JAX_PLATFORMS"] == "cuda"
assert configuration["jax_platforms"] == "cuda"
assert configuration["jax_enable_x64"] is True
assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
assert events[-1] == "preflight"
assert "jax-cuda13-plugin" not in installed and "jax-cuda13-pjrt" not in installed
assert installed["jax-cuda12-plugin"] == installed["jax-cuda12-pjrt"] == "0.5.3"
'''
    execute(program, subprocess_env(tmp_path, backend))
