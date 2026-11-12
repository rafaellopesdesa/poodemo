"""Exercise notebook 09's actual setup with a stale Colab checkout and imports."""
import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]


def notebook_cells():
    notebook = json.loads((REPO / 'notebooks/09_reference_ratio_spline_templates.ipynb').read_text())
    return {cell['id']: ''.join(cell['source']) for cell in notebook['cells']}


@pytest.mark.parametrize('download_fails', [False, True])
def test_fresh_download_replaces_cached_modules_without_touching_drive(tmp_path, download_fails):
    cells = notebook_cells()
    bootstrap = next(source for source in cells.values() if 'RUN_NAME =' in source)
    content = tmp_path / 'content'
    content.mkdir()
    bootstrap = bootstrap.replace('/content', str(content))
    source_commit = next(ast.literal_eval(node.value) for node in ast.parse(bootstrap).body
                         if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'SOURCE_COMMIT')
    check = cells['spline-source-check']
    # Exercise the real archive handling, path activation, module clearing and
    # early check. Stub only Colab, installation and the HTTP response.
    program = f'''
import io
import os
from pathlib import Path
import sys
import tarfile
import types
import urllib.error
import urllib.request

content = Path({str(content)!r})
old = content / "poodemo"
(old / "poodemo").mkdir(parents=True)
(old / "pyproject.toml").write_text("old checkout")
(old / "poodemo/__init__.py").write_text("")
(old / "poodemo/reference_splines.py").write_text("METHODS = ('bin_pchip', 'cdf_z', 'cdf_u')")
(old / "poodemo/pipeline.py").write_text("VERSION = 'old'")
(content / "poodemo_source.zip").write_bytes(b"stale uploaded archive")
drive = content / "drive/run"
drive.mkdir(parents=True)
(drive / "models.bin").write_bytes(b"saved models")
sys.path.insert(0, str(old))
import poodemo.reference_splines
import poodemo.pipeline
stale_module = poodemo.reference_splines

archive_bytes = io.BytesIO()
with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
    for name, source in {{
        "pyproject.toml": "[project]",
        "poodemo/__init__.py": "",
        "poodemo/reference_splines.py": "METHODS = ('ecdf_u',); STUDY_VERSION = 5",
        "poodemo/pipeline.py": "VERSION = 'new'",
    }}.items():
        data = source.encode()
        entry = tarfile.TarInfo("download/" + name)
        entry.size = len(data)
        archive.addfile(entry, io.BytesIO(data))
downloads = []
def urlopen(request, timeout):
    downloads.append(request.full_url)
    assert request.full_url.endswith('/tarball/{source_commit}')
    if {download_fails!r}:
        raise urllib.error.HTTPError(request.full_url, 500, "unavailable", {{}}, None)
    return io.BytesIO(archive_bytes.getvalue())
urllib.request.urlopen = urlopen
google = types.ModuleType("google")
google.__path__ = []
colab = types.ModuleType("google.colab")
colab.drive = types.SimpleNamespace(mount=lambda *args, **kwargs: None)
google.colab = colab
sys.modules.update({{"google": google, "google.colab": colab}})
os.environ["POODEMO_SKIP_INSTALL"] = "1"
os.environ["POODEMO_JAX_BACKEND"] = "cpu"
os.environ["POODEMO_ROOT"] = str(drive)
namespace = {{}}
if {download_fails!r}:
    try:
        exec({bootstrap!r}, namespace)
    except RuntimeError as error:
        assert 'Repository download failed' in str(error)
    else:
        raise AssertionError('Failed downloads must not fall back to the stale source')
else:
    exec({bootstrap!r}, namespace)
    exec({check!r}, namespace)
    first_checkout = namespace['REPO_DIR']
    first_module = namespace['_reference_splines']
    assert first_checkout != old
    assert first_module is not stale_module
    import poodemo.pipeline
    assert poodemo.pipeline.VERSION == 'new'
    # A second Run all in the same interpreter must refresh again.
    exec({bootstrap!r}, namespace)
    exec({check!r}, namespace)
    assert namespace['REPO_DIR'] != first_checkout
    assert namespace['_reference_splines'] is not first_module
    assert len(downloads) == 2
assert (old / 'pyproject.toml').read_text() == 'old checkout'
assert (content / 'poodemo_source.zip').read_bytes() == b'stale uploaded archive'
assert (drive / 'models.bin').read_bytes() == b'saved models'
'''
    completed = subprocess.run([sys.executable, '-I', '-c', program], cwd=content,
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_source_check_rejects_old_methods_before_training_or_fitting(monkeypatch):
    old = SimpleNamespace(METHODS=('bin_pchip', 'cdf_z', 'cdf_u'))
    package = SimpleNamespace(reference_splines=old)
    monkeypatch.setitem(sys.modules, 'poodemo', package)
    monkeypatch.setitem(sys.modules, 'poodemo.reference_splines', old)
    with pytest.raises(RuntimeError, match='Old interpolation code'):
        exec(notebook_cells()['spline-source-check'], {'IN_COLAB': False})


def test_builder_preserves_multiline_user_configuration(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / 'scripts'))
    from build_reference_spline_notebook import fresh_bootstrap
    source = next(s for s in notebook_cells().values() if 'RUN_NAME =' in s)
    source = source.replace('CONFIG_OVERRIDES = {}',
                            'CONFIG_OVERRIDES = {\n    "quadrature_per_process": 12345,\n}')
    updated = fresh_bootstrap(source)
    assert '"quadrature_per_process": 12345' in updated
    assert fresh_bootstrap(updated) == updated
