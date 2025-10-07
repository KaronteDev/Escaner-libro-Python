import os, sys
sys.path.append(r'M:\Trabajo Privado\FAT\Escaner libro Python')
import cv2
import numpy as np
import importlib
mod = importlib.import_module('book_scanner')
from book_scanner import detectar_libro, four_point_transform
img_path = r'C:\Users\Recursos\Pictures\Camera Roll\WIN_20251008_00_07_03_Pro.jpg'
outd = r'M:\Trabajo Privado\FAT\Escaner libro Python\output\temp_test'
os.makedirs(outd, exist_ok=True)
img = cv2.imread(img_path)
if img is None:
    print('ERROR: failed to load image', img_path)
    raise SystemExit(1)
pts = detectar_libro(img)
print('detected pts:', pts)
vis = img.copy()
if pts is not None:
    try:
        cv2.polylines(vis, [np.int32(pts)], True, (0,255,0), 4)
        warped = four_point_transform(img, pts)
        cv2.imwrite(os.path.join(outd, 'warped.jpg'), warped)
        print('warped saved')
    except Exception as e:
        print('error warping:', e)
cv2.imwrite(os.path.join(outd, 'preview.jpg'), vis)
print('preview saved to', outd)
