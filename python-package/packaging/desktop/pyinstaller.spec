# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

block_cipher = None

project_root = Path.cwd()
icon_dir = project_root / "insightface" / "gui" / "assets"
runtime_icon_datas = [
    (str(icon_dir / name), "insightface/gui/assets")
    for name in ("app_icon.svg", "app_icon.png", "app_icon.ico", "app_icon.icns")
    if (icon_dir / name).exists()
]
privateframe_config_dir = (
    project_root / "insightface" / "app" / "privateframe" / "configs"
)
privateframe_config_datas = [
    (str(config_path), "insightface/app/privateframe/configs")
    for config_path in sorted(privateframe_config_dir.glob("*.yaml"))
]
if not privateframe_config_datas:
    raise FileNotFoundError(
        f"PrivateFrame configuration files were not found in {privateframe_config_dir}"
    )

windows_icon = icon_dir / "app_icon.ico"
macos_icon = icon_dir / "app_icon.icns"
linux_icon = icon_dir / "app_icon.png"
exe_icon = None
if sys.platform.startswith("win") and windows_icon.exists():
    exe_icon = str(windows_icon)
elif sys.platform == "darwin" and macos_icon.exists():
    exe_icon = str(macos_icon)
elif linux_icon.exists():
    exe_icon = str(linux_icon)

hiddenimports = [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "onnxruntime",
    "cv2",
    "PIL.Image",
    "sklearn.cluster",
    "reportlab.pdfgen.canvas",
    "insightface.gui.__main__",
    "insightface.gui.app",
    "insightface.gui.main_window",
]

datas = runtime_icon_datas + privateframe_config_datas
binaries = []

a = Analysis(
    ["packaging/desktop/pyinstaller_entry.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="InsightFace Evaluation Studio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=exe_icon,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="InsightFace Evaluation Studio",
)

app = BUNDLE(
    coll,
    name="InsightFace Evaluation Studio.app",
    icon=str(macos_icon) if macos_icon.exists() else None,
    bundle_identifier="ai.insightface.evaluationstudio",
    info_plist={
        "CFBundleName": "InsightFace Evaluation Studio",
        "CFBundleDisplayName": "InsightFace Evaluation Studio",
        "CFBundleIdentifier": "ai.insightface.evaluationstudio",
        "CFBundleShortVersionString": "2.0",
        "CFBundleVersion": "2.0",
        "NSHighResolutionCapable": True,
    },
)
