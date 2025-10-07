# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['book_scanner.py'],                # Archivo principal
    pathex=['.'],               # Carpeta base del proyecto
    binaries=[],
    datas=[('output', 'output')],  # Carpeta de salida de imágenes
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BookScanner',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # False = ventana sin consola
    icon='icono.ico',       # Aquí coloca tu icono
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BookScanner'
)
