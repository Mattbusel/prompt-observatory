# PyInstaller spec for the single-file prompt-observatory executable.
#
#   python -m PyInstaller --noconfirm packaging/prompt-observatory.spec
#
# Run from the repository root. If a ./tiktoken_cache folder exists (the
# release workflow fills it), the tokenizer encodings ship inside the binary
# so token counting works offline.
import os

from PyInstaller.utils.hooks import collect_data_files

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))  # noqa: F821 (SPECPATH is injected)

datas = []
for package in ("gradio", "gradio_client", "safehttpx", "groovy"):
    try:
        datas += collect_data_files(package)
    except Exception:
        pass  # optional in some Gradio versions

cache = os.path.join(ROOT, "tiktoken_cache")
if os.path.isdir(cache):
    datas.append((cache, "tiktoken_cache"))

a = Analysis(  # noqa: F821
    [os.path.join(ROOT, "observatory", "__main__.py")],
    pathex=[ROOT],
    datas=datas,
    hiddenimports=["tiktoken_ext", "tiktoken_ext.openai_public"],
    excludes=["torch", "tensorflow", "matplotlib.tests", "numpy.tests"],
    # Gradio reads its own .py sources at import time, so they must ship as
    # real files next to the compiled modules.
    module_collection_mode={"gradio": "pyz+py", "gradio_client": "pyz+py"},
)
pyz = PYZ(a.pure)  # noqa: F821
exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="prompt-observatory",
    console=True,
    upx=False,
)
