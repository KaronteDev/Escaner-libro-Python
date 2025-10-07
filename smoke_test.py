import importlib, sys, os
sys.path.append(r'M:\Trabajo Privado\FAT\Escaner libro Python')
try:
    m = importlib.import_module('book_scanner')
    print('IMPORT_OK')
    from book_scanner import detectar_libro, four_point_transform, mesh_unwarp
    samples = [
        r'M:\Trabajo Privado\FAT\Escaner libro Python\aa\aaa_001_p1.png',
        r'M:\Trabajo Privado\FAT\Escaner libro Python\bb\bbbb_001_p1.png',
        r'M:\Trabajo Privado\FAT\Escaner libro Python\cc\cc_001_p1.png',
    ]
    import cv2
    for sample in samples:
        if not os.path.exists(sample):
            print('sample not found:', sample)
            continue
        img = cv2.imread(sample)
        if img is None:
            print('Could not read sample image (cv2.imread returned None):', sample)
            continue
        print('\n--- Testing', sample)
        pts = detectar_libro(img)
        print('detected pts:', None if pts is None else pts.shape)
        if pts is not None:
            proc = four_point_transform(img, pts)
            print('four_point_transform ->', proc.shape)
            uw = mesh_unwarp(proc, intensity=0.9)
            print('mesh_unwarp ->', uw.shape)
except Exception as e:
    print('IMPORT_ERR', e)
    raise
