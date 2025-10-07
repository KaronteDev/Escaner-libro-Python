import os, sys
sys.path.append(r'M:\Trabajo Privado\FAT\Escaner libro Python')
import cv2
import numpy as np
from PIL import Image
img_path = r'C:\Users\Recursos\Pictures\Camera Roll\WIN_20251008_00_07_03_Pro.jpg'
outd = r'M:\Trabajo Privado\FAT\Escaner libro Python\output\temp_test'
os.makedirs(outd, exist_ok=True)
img = cv2.imread(img_path)
if img is None:
    print('ERROR: failed to load', img_path); raise SystemExit(1)
h, w = img.shape[:2]
print('image shape:', img.shape)
# prepare
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_eq = clahe.apply(gray)
cv2.imwrite(os.path.join(outd,'gray_eq.jpg'), gray_eq)
blur = cv2.GaussianBlur(gray_eq,(5,5),0)
# try multiple canny thresholds
pairs = [(50,150),(75,200),(30,120)]
for (c1,c2) in pairs:
    edges = cv2.Canny(blur,c1,c2)
    cv2.imwrite(os.path.join(outd,f'edges_{c1}_{c2}.jpg'), edges)
# skin mask
try:
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, np.array((0,133,77),np.uint8), np.array((255,173,127),np.uint8))
    cv2.imwrite(os.path.join(outd,'skin_mask.jpg'), skin_mask)
    non_skin = cv2.bitwise_not(skin_mask)
except Exception as e:
    print('skin mask fail', e); non_skin = None
# base edges
edges = cv2.Canny(blur,75,200)
if non_skin is not None:
    try:
        non_skin_bin = (non_skin>0).astype('uint8')*255
        edges = cv2.bitwise_and(edges, non_skin_bin)
    except Exception as e:
        print('bitwise_and fail', e)
# morphology
dk = max(7, int(min(h,w)/80))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(dk,dk))
edges_f = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
cv2.imwrite(os.path.join(outd,'edges_closed.jpg'), edges_f)
contours,_ = cv2.findContours(edges_f, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    print('no contours from edges')
else:
    print('contours found:', len(contours))
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:100]
# diagnostics per contour
cands = []
for i,c in enumerate(contours[:50]):
    area = cv2.contourArea(c)
    if area < 1000:
        continue
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02*peri, True)
    mask = np.zeros(gray_eq.shape, dtype=np.uint8)
    cv2.drawContours(mask, [hull], -1, 255, -1)
    mean_in = float(cv2.mean(gray_eq, mask=mask)[0]) if np.any(mask>0) else 0.0
    _, std_in = cv2.meanStdDev(gray_eq, mask=mask)
    std_in = float(std_in[0][0])
    # ring
    try:
        ring = cv2.bitwise_and(cv2.dilate(mask, kernel, iterations=1), cv2.bitwise_not(mask))
        mean_ring = float(cv2.mean(gray_eq, mask=ring)[0]) if np.any(ring>0) else 0.0
    except Exception:
        mean_ring = 0.0
    contrast = mean_in - mean_ring
    bx,by,bw,bh = cv2.boundingRect(hull)
    ar = float(bw)/float(bh) if bh>0 else 1.0
    expected_ar = 1.4
    ar_score = max(0.0, 1.0 - abs(ar-expected_ar)/expected_ar)
    score = contrast*(area/(w*h)) + 1.0*ar_score - 0.5*std_in
    cands.append({'idx':i,'area':area,'mean_in':mean_in,'std_in':std_in,'mean_ring':mean_ring,'contrast':contrast,'ar':ar,'ar_score':ar_score,'score':score,'pts': (approx.reshape(-1,2) if len(approx)==4 else None)})
# sort and print top candidates
cands_sorted = sorted(cands, key=lambda x: x['score'], reverse=True)
print('top candidates:')
for c in cands_sorted[:8]:
    print(c)
# save image with top candidates drawn
vis = img.copy()
for k,c in enumerate(cands_sorted[:8]):
    idx = c['idx']
    cnt = contours[idx]
    color = (0,255,0) if k==0 else (0,128,255)
    cv2.drawContours(vis, [cnt], -1, color, 2)
    bx,by,bw,bh = cv2.boundingRect(cnt)
    cv2.rectangle(vis, (bx,by),(bx+bw,by+bh), color, 1)
cv2.imwrite(os.path.join(outd,'top_cands.jpg'), vis)
print('wrote diagnostic images to', outd)
