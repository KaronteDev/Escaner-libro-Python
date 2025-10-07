import cv2
import numpy as np
import os
import json
import tkinter as tk
from tkinter import ttk, messagebox, Scrollbar, filedialog
from PIL import Image, ImageTk
import threading
import time
import datetime
from PyPDF2 import PdfReader, PdfWriter
import re


# --- Image utility functions -------------------------------------------------

def ordenar_puntos(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detectar_libro(frame, canny1=75, canny2=200, min_area=10000, aspect_weight=1.0, use_bright_fallback=True, debug=False):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None
    # improve contrast locally
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
    except Exception:
        gray_eq = gray

    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    edges = cv2.Canny(blur, canny1, canny2)

    # attempt to remove skin/fingers from edges (simple YCrCb threshold)
    try:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        # common skin color range in YCrCb (may vary). This masks likely skin regions.
        skin_mask = cv2.inRange(ycrcb, np.array((0, 133, 77), dtype=np.uint8), np.array((255, 173, 127), dtype=np.uint8))
        # invert skin mask so we keep non-skin areas
        non_skin = cv2.bitwise_not(skin_mask)
        # make non_skin binary 0/255 and combine with edges
        try:
            non_skin_bin = (non_skin > 0).astype('uint8') * 255
            edges = cv2.bitwise_and(edges, non_skin_bin)
        except Exception:
            pass
    except Exception:
        pass

    # morphological closing to fill gaps and smooth small protrusions
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    except Exception:
        pass

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # fallback: try bright-region detection (page tends to be brighter than background)
        try:
            # use adaptive threshold to isolate bright areas
            th = cv2.adaptiveThreshold(gray_eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, -10)
            # invert so bright areas are white
            th_inv = cv2.bitwise_not(th)
            # morphological open to remove small noise
            mk = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            thf = cv2.morphologyEx(th_inv, cv2.MORPH_OPEN, mk)
            contours2, _ = cv2.findContours(thf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours2:
                contours = sorted(contours2, key=cv2.contourArea, reverse=True)[:12]
            else:
                return None
        except Exception:
            return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:32]
    candidates = []
    h_img, w_img = gray_eq.shape[:2]
    # adaptive thresholds per-image (robust to lighting): derive from IQR and global std
    try:
        iqr = float(np.percentile(gray_eq, 75) - np.percentile(gray_eq, 25))
        global_std = float(np.std(gray_eq))
    except Exception:
        iqr = 30.0
        global_std = 50.0
    # convert to sensible thresholds (guard with minima)
    contrast_thresh = max(8.0, iqr * 0.12)
    std_thresh = max(30.0, global_std * 1.2)
    # adaptive kernel for ring dilation
    dk = max(7, int(min(h_img, w_img) / 80))
    try:
        ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))
    except Exception:
        ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    for c in contours:
        try:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            # smooth contour using convex hull to remove finger-like protrusions
            try:
                hull = cv2.convexHull(c)
            except Exception:
                hull = c
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

            # build mask for this candidate
            mask = np.zeros(gray_eq.shape, dtype=np.uint8)
            try:
                cv2.drawContours(mask, [hull], -1, 255, -1)
            except Exception:
                try:
                    cv2.drawContours(mask, [c], -1, 255, -1)
                except Exception:
                    continue

            # compute interior mean and stddev — use an eroded inner mask to avoid dark text/edges
            try:
                # erosion to get a central area (reduce influence of margins and text)
                ek = max(3, int(min(h_img, w_img) / 200))
                inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek)), iterations=2)
                if np.any(inner > 0):
                    mean_in = float(cv2.mean(gray_eq, mask=inner)[0])
                    _, std_in = cv2.meanStdDev(gray_eq, mask=inner)
                    std_in = float(std_in[0][0])
                else:
                    mean_in = float(cv2.mean(gray_eq, mask=mask)[0])
                    _, std_in = cv2.meanStdDev(gray_eq, mask=mask)
                    std_in = float(std_in[0][0])
            except Exception:
                mean_in = float(np.mean(gray_eq[mask > 0])) if np.any(mask > 0) else 0.0
                std_in = float(np.std(gray_eq[mask > 0])) if np.any(mask > 0) else 255.0

            # compute ring mean (dilated mask minus eroded inner area) — sample immediate surroundings
            try:
                dil = cv2.dilate(mask, ring_kernel, iterations=1)
                # exclude interior center if available
                try:
                    inner = inner  # from above
                except Exception:
                    inner = None
                if inner is not None and np.any(inner > 0):
                    ring = cv2.bitwise_and(dil, cv2.bitwise_not(inner))
                else:
                    ring = cv2.bitwise_and(dil, cv2.bitwise_not(mask))
                mean_ring = float(cv2.mean(gray_eq, mask=ring)[0]) if np.any(ring > 0) else 0.0
            except Exception:
                mean_ring = 0.0

            # contrast: page tends to be brighter than surroundings
            contrast = mean_in - mean_ring

            # heuristics: prefer large areas with positive contrast and low interior variance
            # (contrast_thresh and std_thresh adapted per-image above)

            # compute candidate pts (prefer polygon if 4 vertices, fallback to extremes)
            pts = None
            if len(approx) == 4:
                pts = approx.reshape(4, 2)
            else:
                try:
                    pts_all = hull.reshape(-1, 2)
                    sums = pts_all.sum(axis=1)
                    diffs = np.diff(pts_all, axis=1).reshape(-1)
                    tl = pts_all[np.argmin(sums)]
                    br = pts_all[np.argmax(sums)]
                    tr = pts_all[np.argmin(diffs)]
                    bl = pts_all[np.argmax(diffs)]
                    pts = np.vstack([tl, tr, br, bl])
                except Exception:
                    pts = None

            # aspect ratio heuristic: pages usually have a tall-ish rectangle (e.g. ~1.2-1.6)
            try:
                bx, by, bw, bh = cv2.boundingRect(hull)
                if bh > 0:
                    ar = float(bw) / float(bh)
                else:
                    ar = 1.0
            except Exception:
                ar = 1.0
            # expected aspect ratio (width/height) for a page; tuned for typical book pages
            expected_ar = 1.4
            # ar_score in [0,1], higher when close to expected_ar
            ar_score = max(0.0, 1.0 - abs(ar - expected_ar) / expected_ar)

            # scoring: reward contrast and area, include aspect ratio term, penalize high internal variance
            score = contrast * (area / (w_img * h_img)) + (float(aspect_weight) * ar_score) - 0.5 * std_in

            # require minimal criteria: positive contrast and not too noisy
            area_frac = float(area) / float(w_img * h_img)
            if contrast >= contrast_thresh and std_in <= std_thresh and pts is not None:
                candidates.append((score, pts))
            else:
                # keep as lower-priority candidate if area is huge and pts present
                if pts is not None and area > (w_img * h_img * 0.5):
                    candidates.append((score * 0.5, pts))
                else:
                    # fallback: sometimes reflections or lighting make the ring brighter than the page
                    # accept candidates that are reasonably bright, large enough, and have a good aspect-ratio match
                    if pts is not None and mean_in >= 120.0 and area_frac >= 0.12 and ar_score >= 0.6 and std_in <= (std_thresh * 1.6):
                        candidates.append((score * 0.7, pts))
                    else:
                        continue
        except Exception:
            continue

    if candidates:
        # choose best-scoring candidate
        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0][1]
        if debug:
            # build a summary list for debugging: score, area fraction, aspect ratio, contrast
            dbg = []
            for sc, pts in candidates:
                try:
                    # compute bounding rect and area
                    bx, by, bw, bh = cv2.boundingRect(np.array(pts, dtype=np.int32))
                    area = bw * bh
                    ar = float(bw) / float(bh) if bh > 0 else 1.0
                except Exception:
                    area = 0
                    ar = 1.0
                dbg.append({'score': float(sc), 'area': int(area), 'ar': float(ar), 'pts': pts.tolist() if hasattr(pts, 'tolist') else pts})
            return best, dbg
        return best
    # fallback: try finding a large bright connected region (useful when ring/contrast heuristics fail)
    try:
        # Otsu threshold to separate bright page from darker background
        _, th = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # morphological open/close to remove noise and small text holes
        mk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, int(min(h_img, w_img)/150)),) * 2)
        thf = cv2.morphologyEx(th, cv2.MORPH_OPEN, mk)
        thf = cv2.morphologyEx(thf, cv2.MORPH_CLOSE, mk)
        contours2, _ = cv2.findContours(thf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours2:
            contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)
            for c in contours2[:6]:
                a = cv2.contourArea(c)
                if a < (w_img * h_img * 0.03):
                    continue
                bx, by, bw, bh = cv2.boundingRect(c)
                ar = float(bw) / float(bh) if bh > 0 else 1.0
                # Accept if aspect ratio within 0.6..1.8 (covers landscape/portrait small skew)
                if 0.6 <= (ar if ar>0 else 1.0) <= 1.8:
                    # try to approximate polygon to 4 points
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                    if len(approx) == 4:
                        pts = approx.reshape(4, 2)
                    else:
                        pts = np.array([[bx,by],[bx+bw,by],[bx+bw,by+bh],[bx,by+bh]], dtype=np.float32)
                    if debug:
                        return pts, [{'score': float(a), 'area': int(a), 'ar': float(ar), 'pts': pts.tolist()}]
                    return pts
    except Exception:
        pass
    return None
    return None


def four_point_transform(image, pts):
    rect = ordenar_puntos(pts)
    (tl, tr, br, bl) = rect
    anchoA = np.linalg.norm(br - bl)
    anchoB = np.linalg.norm(tr - tl)
    maxAncho = int(max(anchoA, anchoB))
    altoA = np.linalg.norm(tr - br)
    altoB = np.linalg.norm(tl - bl)
    maxAlto = int(max(altoA, altoB))
    # correct destination ordering: tl, tr, br, bl
    dst = np.array([[0, 0], [maxAncho - 1, 0], [maxAncho - 1, maxAlto - 1], [0, maxAlto - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxAncho, maxAlto))


def separar_paginas(imagen):
    h, w = imagen.shape[:2]
    mitad = w // 2
    return imagen[:, :mitad], imagen[:, mitad:]


def mesh_unwarp(image, intensity=0.8):
    """Perform a mesh-based horizontal unwarp to reduce book curvature.

    Parameters:
    - image: BGR numpy array
    - intensity: float, how strong the unwarp should be (0.0 no-op, ~1.0 default)

    This builds a per-column displacement field (vectorized) and applies cv2.remap.
    """
    try:
        if intensity is None or intensity <= 0:
            return image
        h, w = image.shape[:2]
        # grayscale projection to find dark spine region
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cx = w // 2
            strip = gray[:, max(0, cx - w // 6): min(w, cx + w // 6)]
            v = np.mean(strip.astype(np.float32), axis=0)
            min_idx = int(np.argmin(v))
            spine_x = max(0, cx - w // 6 + min_idx)
        except Exception:
            spine_x = w // 2

        # normalized coordinate grid
        xs = np.arange(w, dtype=np.float32)
        # distance from spine normalized [-1,1]
        dx = (xs - spine_x) / float(w)

        # profile: cubic-like displacement that is zero at edges and max near spine
        # amplitude proportional to width and intensity
        max_shift = float(w) * 0.06 * float(np.clip(intensity, 0.0, 3.0))
        # use a smooth cubic curve: shift = sign(dx) * (1 - (1-|dx|)^3) * max_shift
        ad = np.abs(dx)
        profile = (1.0 - np.power(1.0 - np.clip(ad, 0.0, 1.0), 3.0))
        shift_x = -np.sign(dx) * profile * max_shift

        # create full map by repeating per row; optionally taper effect towards top/bottom
        # vertical taper to reduce edge stretching
        ys = np.arange(h, dtype=np.float32)
        vy = 1.0 - 0.3 * ((ys - h/2.0)/(h/2.0))**2  # gentle quadratic taper
        vy = np.clip(vy, 0.6, 1.0)

        map_x = np.empty((h, w), dtype=np.float32)
        map_y = np.empty((h, w), dtype=np.float32)
        for i, vyv in enumerate(vy):
            map_x[i, :] = np.clip(xs + shift_x * vyv, 0, w - 1)
            map_y[i, :] = i

        unwarped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return unwarped
    except Exception:
        return image


def mesh_unwarp_advanced(image, cols=40, intensity=0.8):
    """Advanced mesh remapping: build a coarse mesh of control columns and interpolate per-pixel displacement.

    - cols: number of vertical control columns (coarse). Higher -> more precise but slower.
    - intensity: multiplier for displacement.
    """
    try:
        if intensity is None or intensity <= 0:
            return image
        h, w = image.shape[:2]
        cols = int(max(4, min(cols, w)))

        # estimate spine position
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cx = w // 2
            strip = gray[:, max(0, cx - w // 6): min(w, cx + w // 6)]
            v = np.mean(strip.astype(np.float32), axis=0)
            min_idx = int(np.argmin(v))
            spine_x = max(0, cx - w // 6 + min_idx)
        except Exception:
            spine_x = w // 2

        # control column x positions
        ctrl_x = np.linspace(0, w - 1, num=cols, dtype=np.float32)
        # compute displacement for each control column (same profile idea as mesh_unwarp)
        dx = (ctrl_x - spine_x) / float(w)
        ad = np.abs(dx)
        profile = (1.0 - np.power(1.0 - np.clip(ad, 0.0, 1.0), 3.0))
        max_shift = float(w) * 0.06 * float(np.clip(intensity, 0.0, 3.0))
        shift_ctrl = -np.sign(dx) * profile * max_shift

        # Build per-pixel shift by linear interpolation of control columns
        xs = np.arange(w, dtype=np.float32)
        # interpolate shift_ctrl defined at ctrl_x to every xs
        shift_full = np.interp(xs, ctrl_x, shift_ctrl)

        # vertical taper (reduce near top/bottom)
        ys = np.arange(h, dtype=np.float32)
        vy = 1.0 - 0.4 * ((ys - h/2.0)/(h/2.0))**2
        vy = np.clip(vy, 0.4, 1.0)

        map_x = np.empty((h, w), dtype=np.float32)
        map_y = np.empty((h, w), dtype=np.float32)
        for i, vyv in enumerate(vy):
            map_x[i, :] = np.clip(xs + shift_full * vyv, 0, w - 1)
            map_y[i, :] = i

        unwarped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return unwarped
    except Exception:
        return image


# --- Application --------------------------------------------------------------

class BookScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📚 Escáner de libros con OpenCV")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f8f9fa")

        # Camera state
        self.cam_index = 0
        self.video = None
        self.frame_actual = None
        self.frame_lock = threading.Lock()

        # Scanning state
        self._running = False
        self._capture_thread = None
        self._scanning = False

        # Data/UI
        self.carpeta_salida = ""
        self.contador = 1
        self.thumbnails = []
        # debug flag to help diagnose drag issues
        self._debug_drag = False

        # Form vars
        self.titulo_var = tk.StringVar()
        self.autor_var = tk.StringVar()
        self.tema_var = tk.StringVar()
        self.signatura_var = tk.StringVar()
        self.archivo_var = tk.StringVar()
        self.cam_var = tk.StringVar()

        # Build UI
        self.crear_interfaz()

        # Start camera and capture thread
        self._enumerar_camaras()
        # Load last session (if any) so user can continue working in previous folder
        try:
            self._load_last_session()
        except Exception:
            pass
        self._abrir_camara(self.cam_index)
        self._start_capture_thread()

        # Start UI refresh
        self.actualizar_video()

    def crear_interfaz(self):
        # Left column: create a scrollable area for the form (use a canvas + scrollbar)
        left_outer = ttk.Frame(self.root)
        left_outer.pack(side="left", fill="y", padx=10, pady=10)
        left_canvas = tk.Canvas(left_outer, borderwidth=0, highlightthickness=0, width=360)
        left_scroll = ttk.Scrollbar(left_outer, orient='vertical', command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left_canvas.pack(side='left', fill='y', expand=False)
        left_scroll.pack(side='right', fill='y')
        form_container = ttk.Frame(left_canvas)
        # ensure scrollregion updates
        form_container.bind('<Configure>', lambda e: left_canvas.configure(scrollregion=left_canvas.bbox('all')))
        left_canvas.create_window((0, 0), window=form_container, anchor='nw')

        frame_form = ttk.LabelFrame(form_container, text="Datos del documento", padding=10)
        frame_form.pack(side="top", fill="x", padx=0, pady=0)
        self.frame_form = frame_form

        # helper: add collapse/expand toggle to a LabelFrame by wrapping header
        def make_collapsible(parent, title=None):
            # title is optional so callers may create the frame and configure text later
            lf = ttk.LabelFrame(parent, text=(title if title is not None else ""))
            # store a hidden flag
            lf._collapsed = False
            lf._hidden_children = []
            def toggle():
                try:
                    if lf._collapsed:
                        for w, pack_info in getattr(lf, '_hidden_children', []):
                            try:
                                # restore using stored pack options when available
                                w.pack(**(pack_info or {}))
                            except Exception:
                                try:
                                    w.pack()
                                except Exception:
                                    pass
                        lf._hidden_children = []
                        lf._collapsed = False
                        try:
                            lf._btn.configure(text='▾')
                        except Exception:
                            pass
                    else:
                        # hide current children but keep the toggle button visible
                        childs = [c for c in lf.winfo_children() if c is not getattr(lf, '_btn', None)]
                        store = []
                        for w in childs:
                            try:
                                # capture simple pack info if available
                                info = None
                                try:
                                    info = w.pack_info()
                                except Exception:
                                    info = None
                                try:
                                    w.pack_forget()
                                except Exception:
                                    pass
                                store.append((w, info))
                            except Exception:
                                pass
                        lf._hidden_children = store
                        lf._collapsed = True
                        try:
                            lf._btn.configure(text='▸')
                        except Exception:
                            pass
                except Exception:
                    pass
            # small button in top-right to toggle; we'll place it so it stays visible when collapsing
            try:
                btn = ttk.Button(lf, text='▾', width=2, command=toggle)
                lf._btn = btn
                # attempt to place the button anchored to the top-right of the labelframe
                try:
                    # a small offset to keep button inside the border
                    btn.place(in_=lf, relx=1.0, x=-6, y=6, anchor='ne')
                except Exception:
                    # fallback: pack top-right (may appear in child list and be hidden)
                    try:
                        btn.pack(side='right')
                    except Exception:
                        pass
            except Exception:
                pass
            return lf

        # group main controls into labeled sections
        meta_frame = make_collapsible(frame_form)
        meta_frame.configure(text='Metadatos')
        meta_frame.pack(fill='x', pady=(4,6))
        thumb_frame = make_collapsible(frame_form)
        thumb_frame.configure(text='Miniaturas')
        thumb_frame.pack(fill='x', pady=(4,6))
        det_frame = make_collapsible(frame_form)
        det_frame.configure(text='Detección y limpieza')
        det_frame.pack(fill='x', pady=(4,6))
        cam_frame = make_collapsible(frame_form)
        cam_frame.configure(text='Cámara')
        cam_frame.pack(fill='x', pady=(4,6))

        # selection style for thumbnails
        try:
            style = ttk.Style(self.root)
            # define selected frame style with a visible border / background
            style.configure('Selected.TFrame', background='#d0e7ff')
            style.configure('Selected.TLabel', background='#d0e7ff')
        except Exception:
            pass

        ttk.Label(meta_frame, text="Título del libro:").pack(anchor="w")
        ttk.Entry(meta_frame, textvariable=self.titulo_var).pack(fill="x")

        ttk.Label(meta_frame, text="Autor:").pack(anchor="w", pady=(10, 0))
        ttk.Entry(meta_frame, textvariable=self.autor_var).pack(fill="x")

        ttk.Label(meta_frame, text="Tema investigación:").pack(anchor="w", pady=(10, 0))
        ttk.Entry(meta_frame, textvariable=self.tema_var).pack(fill="x")

        ttk.Label(meta_frame, text="Signatura:").pack(anchor="w", pady=(10, 0))
        ttk.Entry(meta_frame, textvariable=self.signatura_var).pack(fill="x")

        ttk.Label(meta_frame, text="Archivo/Biblioteca:").pack(anchor="w", pady=(10, 0))

        ttk.Entry(meta_frame, textvariable=self.archivo_var).pack(fill="x")
        
        # store buttons so we can enable/disable them during scanning
        self.btn_crear = ttk.Button(meta_frame, text="📁 Crear carpeta", command=self.crear_carpeta)
        self.btn_crear.pack(fill="x", pady=(10, 0))
        # Open existing folder to continue working
        self.btn_abrir = ttk.Button(meta_frame, text="📂 Abrir carpeta", command=self.abrir_carpeta)
        self.btn_abrir.pack(fill="x", pady=(6, 0))
        self.btn_export = ttk.Button(meta_frame, text="🧾 Exportar a PDF", command=self.exportar_pdf)
        self.btn_export.pack(fill="x", pady=(6, 0))
        self.btn_reiniciar = ttk.Button(meta_frame, text="🔁 Reiniciar cámara", command=self.reiniciar_camara)
        self.btn_reiniciar.pack(fill="x", pady=(6, 0))
        self.btn_salir = ttk.Button(meta_frame, text="❌ Salir", command=self.salir)
        self.btn_salir.pack(fill="x", pady=(6, 0))
        
        
        # Auto-save metadata option
        self.autosave_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(meta_frame, text="Auto-guardar metadatos", variable=self.autosave_var).pack(fill="x", pady=(6, 0))

        # Thumbnail size controls
        ttk.Label(thumb_frame, text="Tamaño miniatura (px):").pack(anchor="w", pady=(8, 0))
        self.thumb_w_var = tk.IntVar(value=120)
        self.thumb_h_var = tk.IntVar(value=160)
        size_frame = ttk.Frame(thumb_frame)
        size_frame.pack(fill="x")
        tk.Spinbox(size_frame, from_=60, to=400, textvariable=self.thumb_w_var, width=6).pack(side="left")
        ttk.Label(size_frame, text="x").pack(side="left", padx=4)
        tk.Spinbox(size_frame, from_=60, to=400, textvariable=self.thumb_h_var, width=6).pack(side="left")
        ttk.Button(size_frame, text="Aplicar", command=lambda: (self._load_existing_thumbnails())).pack(side="left", padx=8)
        # Detection tuning (in detection frame)
        self.skin_thresh_var = tk.IntVar(value=25)
        ttk.Label(det_frame, text="Umbral piel (Cr range):").pack(anchor='w')
        ttk.Scale(det_frame, from_=10, to=60, variable=self.skin_thresh_var, orient='horizontal').pack(fill='x')
        self.morph_kernel_var = tk.IntVar(value=9)
        ttk.Label(det_frame, text="Kernel morfología: ").pack(anchor='w')
        ttk.Scale(det_frame, from_=3, to=31, variable=self.morph_kernel_var, orient='horizontal').pack(fill='x')
        self.min_area_var = tk.IntVar(value=10000)
        ttk.Label(det_frame, text="Área mínima contorno:").pack(anchor='w')
        tk.Spinbox(det_frame, from_=1000, to=200000, increment=500, textvariable=self.min_area_var).pack(fill='x')
        # aspect ratio weight for scoring
        self.aspect_ratio_weight = tk.DoubleVar(value=1.0)
        ttk.Label(det_frame, text="Peso heurística proporción (aspect ratio):").pack(anchor='w', pady=(6,0))
        ttk.Scale(det_frame, from_=0.0, to=3.0, variable=self.aspect_ratio_weight, orient='horizontal').pack(fill='x')
        # bright-region fallback toggle
        self.use_bright_fallback_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(det_frame, text='Usar fallback región brillante', variable=self.use_bright_fallback_var).pack(anchor='w', pady=(6,0))
        # debug overlay controls
        self.debug_det_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(det_frame, text='Debug detección (mostrar scores)', variable=self.debug_det_var).pack(anchor='w', pady=(6,0))
        self.debug_topn_var = tk.IntVar(value=4)
        ttk.Label(det_frame, text='Top N candidatos a mostrar:').pack(anchor='w')
        tk.Spinbox(det_frame, from_=1, to=12, textvariable=self.debug_topn_var, width=6).pack(anchor='w')
        # curvature correction
        self.curvature_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(det_frame, text="Corregir curvatura (experimental)", variable=self.curvature_var).pack(fill='x', pady=(6,0))
        # curvature intensity slider
        self.curvature_intensity_var = tk.DoubleVar(value=0.8)
        ttk.Label(det_frame, text="Intensidad curvatura:").pack(anchor='w')
        ttk.Scale(det_frame, from_=0.0, to=2.0, variable=self.curvature_intensity_var, orient='horizontal').pack(fill='x')
        # mesh density for advanced unwarp (number of control columns)
        self.curvature_mesh_cols_var = tk.IntVar(value=40)
        ttk.Label(det_frame, text="Densidad malla (columnas):").pack(anchor='w')
        tk.Spinbox(det_frame, from_=8, to=200, increment=2, textvariable=self.curvature_mesh_cols_var).pack(fill='x')
        # Optionally rename files to preserve order when reordering
        self.rename_on_reorder_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(thumb_frame, text="Renombrar archivos al reordenar", variable=self.rename_on_reorder_var).pack(fill="x", pady=(6, 0))

        # Undo last rename (backup) button
        self.btn_undo_rename = ttk.Button(meta_frame, text="↶ Deshacer renombrado", command=self._undo_last_rename)
        self.btn_undo_rename.pack(fill="x", pady=(6, 0))

        # .tmp policy control (ask / commit / delete)
        ttk.Label(det_frame, text="Política archivos .tmp:").pack(anchor='w', pady=(8,0))
        self.tmp_policy_var = tk.StringVar(value='Preguntar')
        self.tmp_policy_combo = ttk.Combobox(det_frame, textvariable=self.tmp_policy_var, state='readonly', values=['Preguntar','Realizar','Borrar'])
        self.tmp_policy_combo.pack(fill='x')

        # Last-saved indicator
        self._last_saved_var = tk.StringVar(value="Metadatos guardados: -")
        ttk.Label(meta_frame, textvariable=self._last_saved_var, foreground="#2e7d32").pack(fill="x", pady=(4, 0))

        # Camera selector (placed inside cam_frame)
        frame_cam = ttk.Frame(cam_frame)
        frame_cam.pack(fill="x", pady=(4, 0))
        # use grid inside this small frame so we can place the capture button below the combobox
        lbl_cam = ttk.Label(frame_cam, text="Cámara:")
        lbl_cam.grid(row=0, column=0, sticky='w')
        self.cam_selector = ttk.Combobox(frame_cam, textvariable=self.cam_var, state="readonly", width=30)
        self.cam_selector.grid(row=0, column=1, sticky='w', padx=(5, 0))
        self.cam_selector.bind('<<ComboboxSelected>>', lambda e: self.cambiar_camara())
        # Capture button placed under the combobox; also expose as self.btn_capturar for _set_ui_enabled
        self.btn_capturar = ttk.Button(frame_cam, text="📸 Capturar", command=self.escanear,width=30)
        self.btn_capturar.grid(row=1, column=0, columnspan=4, pady=(6, 0), sticky='w')
        # BooleanVar doesn't have layout methods; create the variable and place
        # the Checkbutton in the grid instead.
        self.split_var = tk.BooleanVar(frame_cam, value=True)
        ttk.Checkbutton(frame_cam, text="Partir en 2", variable=self.split_var,width=30).grid(row=2, column=0, columnspan=2, pady=(6, 0), sticky='w')

        # Right: video + gallery
        frame_video = ttk.Frame(self.root)
        frame_video.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.lbl_video = ttk.Label(frame_video)
        self.lbl_video.pack(side="top", expand=True)

        # Status indicator
        self.status_var = tk.StringVar(value="Cámara: - | Resolución: - | min: - max: -")
        self.lbl_status = ttk.Label(frame_video, textvariable=self.status_var, anchor="e")
        self.lbl_status.pack(side="top", fill="x")

        frame_gallery = ttk.LabelFrame(frame_video, text="📄 Páginas escaneadas", height=200)
        frame_gallery.pack(side="bottom", fill="x", pady=(10, 0))
        # progress bar for thumbnail loading
        self.thumb_progress = ttk.Progressbar(frame_gallery, orient='horizontal', mode='determinate', maximum=1, value=0)
        self.thumb_progress.pack(side='top', fill='x', pady=(4, 4))
        self.thumb_progress.pack_forget()
        self.canvas_gallery = tk.Canvas(frame_gallery, height=180)
        self.scroll_gallery = Scrollbar(frame_gallery, orient="horizontal", command=self.canvas_gallery.xview)
        self.canvas_gallery.configure(xscrollcommand=self.scroll_gallery.set)
        self.scroll_gallery.pack(side="bottom", fill="x")
        self.canvas_gallery.pack(side="top", fill="x")
        self.frame_thumbs = ttk.Frame(self.canvas_gallery)
        self.canvas_gallery.create_window((0, 0), window=self.frame_thumbs, anchor="nw")
        self.frame_thumbs.bind("<Configure>", lambda e: self.canvas_gallery.configure(scrollregion=self.canvas_gallery.bbox("all")))

        self.root.bind("<space>", lambda e: self.escanear())


        # keyboard navigation for thumbnails
        try:
            self.root.bind('<Right>', lambda e: self._select_next_thumb(e))
            self.root.bind('<Left>', lambda e: self._select_prev_thumb(e))
        except Exception:
            pass
        # Enter to open, Delete to remove selected thumbnail
        try:
            self.root.bind('<Return>', lambda e: self._open_selected_preview())
            self.root.bind('<Delete>', lambda e: self._delete_selected_thumb())
        except Exception:
            pass

        # bind middle mouse button on preview to capture (Button-2 on Windows)
        try:
            self.lbl_video.bind('<Button-2>', lambda e: self.escanear())
        except Exception:
            pass

        # install traces to auto-save metadata on change (immediate save)
        try:
            for v in (self.titulo_var, self.autor_var, self.tema_var, self.signatura_var, self.archivo_var):
                # remove existing traces if present
                try:
                    v.trace_vdelete('w', v._autosave_trace_id)
                except Exception:
                    pass
                tid = v.trace_add('write', lambda *a, vv=v: (self._save_folder_metadata() if getattr(self, 'autosave_var', None) and self.autosave_var.get() else None))
                try:
                    v._autosave_trace_id = tid
                except Exception:
                    pass
        except Exception:
            pass

    def _clear_thumbnails(self):
        try:
            for w in list(self.thumbnails):
                try:
                    w.destroy()
                except Exception:
                    pass
            self.thumbnails.clear()
            # also clear any children inside frame_thumbs
            try:
                for child in list(self.frame_thumbs.winfo_children()):
                    child.destroy()
            except Exception:
                pass
            # preserve any desired order marker (set by _load_folder_metadata) - do not reset here
        except Exception:
            pass

    def _cleanup_tmp_files(self):
        """Scan carpeta_salida for .tmp files left by interrupted operations and offer to recover or delete them."""
        try:
            if not self.carpeta_salida or not os.path.isdir(self.carpeta_salida):
                return
            tmps = [f for f in os.listdir(self.carpeta_salida) if f.endswith('.tmp')]
            if not tmps:
                return
            # decide policy
            policy = getattr(self, 'tmp_policy_var', None) and self.tmp_policy_var.get() or 'Preguntar'
            if policy == 'Borrar':
                for t in tmps:
                    try:
                        os.remove(os.path.join(self.carpeta_salida, t))
                    except Exception:
                        pass
                return
            elif policy == 'Realizar':
                do_commit = True
            else:
                # Ask user what to do
                do_commit = messagebox.askyesno('Archivos temporales encontrados', f'Se han encontrado {len(tmps)} archivos .tmp en la carpeta. ¿Desea intentar completar los renombrados pendientes? (No = eliminar .tmp)')
                if not do_commit:
                    for t in tmps:
                        try:
                            os.remove(os.path.join(self.carpeta_salida, t))
                        except Exception:
                            pass
                    return
            # try to commit tmp -> final (remove .tmp suffix)
            for t in tmps:
                tmp_path = os.path.join(self.carpeta_salida, t)
                final = tmp_path[:-4]
                try:
                    # if final exists, skip and remove tmp
                    if os.path.exists(final):
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                        continue
                    os.replace(tmp_path, final)
                except Exception:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
        except Exception:
            pass

    def _load_existing_thumbnails(self):
        # populate the gallery with existing PNG files in carpeta_salida
        try:
            self._clear_thumbnails()
            if not self.carpeta_salida or not os.path.isdir(self.carpeta_salida):
                return
            archivos = sorted([f for f in os.listdir(self.carpeta_salida) if f.lower().endswith('.png')])

            # if metadata contains an explicit order, honor it: place ordered files first
            try:
                mdpath = self._folder_metadata_path()
                desired = None
                if mdpath and os.path.exists(mdpath):
                    try:
                        with open(mdpath, 'r', encoding='utf-8') as mf:
                            mdata = json.load(mf)
                            desired = mdata.get('order')
                    except Exception:
                        desired = None
                if desired:
                    # keep only files that still exist and end with .png
                    desired = [d for d in desired if d.lower().endswith('.png') and os.path.exists(os.path.join(self.carpeta_salida, d))]
                    remaining = [a for a in archivos if a not in desired]
                    archivos = desired + remaining
                    # store for possible post-load ordering operations
                    self._desired_thumb_order = archivos.copy()
                else:
                    self._desired_thumb_order = None
            except Exception:
                self._desired_thumb_order = None

            # prepare progress
            total = len(archivos)
            try:
                if total > 0:
                    self.thumb_progress.configure(maximum=total, value=0)
                    self.thumb_progress.pack(side='top', fill='x', pady=(4, 4))
                    self._thumb_load_total = total
                    self._thumb_load_done = 0
                else:
                    self.thumb_progress.pack_forget()
            except Exception:
                pass

            # load in background to avoid blocking UI for many images
            def worker(files):
                for fn in files:
                    ruta = os.path.join(self.carpeta_salida, fn)
                    try:
                        # Use PIL in worker to open and resize; create PhotoImage on main thread
                        pil = Image.open(ruta).convert('RGB')
                        w = int(self.thumb_w_var.get())
                        h = int(self.thumb_h_var.get())
                        pil.thumbnail((w, h), Image.LANCZOS)
                        # schedule creation of PhotoImage and widget on main thread
                        try:
                            self.root.after(0, lambda p=pil.copy(), r=ruta: (self._add_thumbnail_from_pil(p, r), self._thumb_progress_step()))
                        except Exception:
                            pass
                    except Exception:
                        continue

            t = threading.Thread(target=worker, args=(archivos,), daemon=True)
            t.start()
        except Exception:
            pass

    def _thumb_progress_step(self):
        try:
            self._thumb_load_done = getattr(self, '_thumb_load_done', 0) + 1
            self.thumb_progress['value'] = self._thumb_load_done
            if getattr(self, '_thumb_load_total', 0) and self._thumb_load_done >= self._thumb_load_total:
                # finished
                try:
                    self.thumb_progress.pack_forget()
                except Exception:
                    pass
                # if metadata requested a particular order, apply it
                try:
                    if getattr(self, '_desired_thumb_order', None):
                        try:
                            self._apply_desired_order()
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass

    def _add_thumbnail_from_pil(self, pil_img, ruta):
        try:
            img_tk = ImageTk.PhotoImage(pil_img)
            # use tk.Frame and tk.Label so we can change bg/relief for selection
            container = tk.Frame(self.frame_thumbs, bd=0, relief='flat')
            lbl_img = tk.Label(container, image=img_tk, bd=0)
            lbl_img.image = img_tk
            # expose image on container so drag ghost can show it
            container.image = img_tk
            lbl_img.pack()
            # filename label (no extension)
            fname = os.path.splitext(os.path.basename(ruta))[0]
            lbl_name = tk.Label(container, text=fname, width=16, anchor='center')
            lbl_name.pack()
            # show full filename on hover via tooltip
            def _enter(e, label=lbl_name, fullpath=ruta):
                try:
                    # show tooltip with full filename (basename)
                    self._show_tooltip(label, os.path.basename(fullpath))
                except Exception:
                    pass

            def _leave(e, label=lbl_name):
                try:
                    self._hide_tooltip(label)
                except Exception:
                    pass

            lbl_name.bind('<Enter>', _enter)
            lbl_name.bind('<Leave>', _leave)
            # attach filepath on container for consistency
            container.filepath = ruta
            container.image_label = lbl_img
            # remember last known image in case fallback is required
            try:
                self.last_known_thumb_image = img_tk
            except Exception:
                pass
            container.name_label = lbl_name
            container.pack(side="left", padx=5, pady=5)
            # double-click opens preview (bind on image label and container)
            lbl_img.bind('<Double-1>', lambda e, r=ruta: self._open_preview(r))
            container.bind('<Double-1>', lambda e, r=ruta: self._open_preview(r))
            # right click shows context menu (bind on container)
            container.bind('<Button-3>', lambda e, c=container: self._show_thumb_menu(e, c))
            # drag-and-drop bindings (bind on container, image and name so clicks on them work)
            container.bind('<ButtonPress-1>', lambda e, c=container: self._on_thumb_press(e, c))
            container.bind('<B1-Motion>', lambda e, c=container: self._on_thumb_motion(e, c))
            container.bind('<ButtonRelease-1>', lambda e, c=container: self._on_thumb_release(e, c))
            lbl_img.bind('<ButtonPress-1>', lambda e, c=container: self._on_thumb_press(e, c))
            lbl_img.bind('<B1-Motion>', lambda e, c=container: self._on_thumb_motion(e, c))
            lbl_img.bind('<ButtonRelease-1>', lambda e, c=container: self._on_thumb_release(e, c))
            lbl_name.bind('<ButtonPress-1>', lambda e, c=container: self._on_thumb_press(e, c))
            lbl_name.bind('<B1-Motion>', lambda e, c=container: self._on_thumb_motion(e, c))
            lbl_name.bind('<ButtonRelease-1>', lambda e, c=container: self._on_thumb_release(e, c))
            self.thumbnails.append(container)
        except Exception:
            pass

    def _show_tooltip(self, widget, text):
        """Show a small tooltip near the widget with the given text."""
        try:
            # hide any existing tooltip for this widget
            try:
                if getattr(widget, '_tooltip_win', None):
                    widget._tooltip_win.destroy()
            except Exception:
                pass
            x = widget.winfo_rootx() + 10
            y = widget.winfo_rooty() + widget.winfo_height() + 4
            win = tk.Toplevel(self.root)
            win.wm_overrideredirect(True)
            win.wm_geometry(f"+{x}+{y}")
            lbl = ttk.Label(win, text=text, background="#ffffe0", relief='solid', borderwidth=1)
            lbl.pack(padx=4, pady=2)
            widget._tooltip_win = win
        except Exception:
            pass

    def _hide_tooltip(self, widget):
        try:
            win = getattr(widget, '_tooltip_win', None)
            if win:
                try:
                    win.destroy()
                except Exception:
                    pass
                widget._tooltip_win = None
        except Exception:
            pass

    def _on_thumb_press(self, event, lbl):
        try:
            self._drag_data = {'widget': lbl, 'start_x': event.x_root, 'start_y': event.y_root}
            self._drag_ghost = None
        except Exception:
            pass

    def _set_selected_thumb(self, lbl):
        """Visually mark a thumbnail as selected and unmark previous."""
        try:
            prev = getattr(self, '_selected_thumb', None)
            if prev is not None and prev is not lbl:
                try:
                    # restore previous look for tk.Frame
                    try:
                        prev.configure(bg=self.frame_thumbs.cget('bg'), relief='flat', bd=0)
                    except Exception:
                        prev.configure(style='TFrame')
                except Exception:
                    try:
                        prev['relief'] = 'flat'
                    except Exception:
                        pass
            # mark new
            try:
                # for tk.Frame change bg and border
                lbl.configure(bg='#d0e7ff', relief='solid', bd=2)
                # also update child labels background
                try:
                    for child in lbl.winfo_children():
                        try:
                            child.configure(bg='#d0e7ff')
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception:
                try:
                    lbl.configure(style='Selected.TFrame')
                except Exception:
                    try:
                        lbl['relief'] = 'solid'
                    except Exception:
                        pass
            self._selected_thumb = lbl
        except Exception:
            pass

    def _start_selection_pulse(self, lbl):
        """Start a small pulsing effect on the selected thumbnail."""
        try:
            # cancel previous
            try:
                prev_id = getattr(self, '_selection_after_id', None)
                if prev_id:
                    self.root.after_cancel(prev_id)
            except Exception:
                pass
            # initialize state
            self._pulse_state = 0
            def _pulse():
                try:
                    if not getattr(self, '_selected_thumb', None) is lbl:
                        return
                    # toggle bd and bg tint
                    try:
                        if getattr(self, '_pulse_state', 0) == 0:
                            lbl.configure(bg='#c6e0ff', bd=3)
                            for child in lbl.winfo_children():
                                try:
                                    child.configure(bg='#c6e0ff')
                                except Exception:
                                    pass
                            self._pulse_state = 1
                        else:
                            lbl.configure(bg='#d0e7ff', bd=2)
                            for child in lbl.winfo_children():
                                try:
                                    child.configure(bg='#d0e7ff')
                                except Exception:
                                    pass
                            self._pulse_state = 0
                    except Exception:
                        pass
                    self._selection_after_id = self.root.after(350, _pulse)
                except Exception:
                    pass
            _pulse()
        except Exception:
            pass

    def _stop_selection_pulse(self, lbl):
        try:
            aid = getattr(self, '_selection_after_id', None)
            if aid:
                try:
                    self.root.after_cancel(aid)
                except Exception:
                    pass
            try:
                # restore look
                lbl.configure(bg=self.frame_thumbs.cget('bg'), relief='flat', bd=0)
                for child in lbl.winfo_children():
                    try:
                        child.configure(bg=self.frame_thumbs.cget('bg'))
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass

    def _open_selected_preview(self):
        try:
            sel = getattr(self, '_selected_thumb', None)
            if not sel:
                return
            ruta = getattr(sel, 'filepath', None)
            if ruta and os.path.exists(ruta):
                self._open_preview(ruta)
        except Exception:
            pass

    def _delete_selected_thumb(self):
        try:
            sel = getattr(self, '_selected_thumb', None)
            if not sel:
                return
            # confirm and delete
            if messagebox.askyesno('Confirmar', '¿Eliminar la miniatura seleccionada?'):
                try:
                    self._delete_thumb(sel)
                    self._selected_thumb = None
                except Exception:
                    pass
        except Exception:
            pass

    def _select_next_thumb(self, event=None):
        try:
            if not self.thumbnails:
                return
            cur = getattr(self, '_selected_thumb', None)
            if cur is None:
                self._set_selected_thumb(self.thumbnails[0])
                return
            try:
                idx = self.thumbnails.index(cur)
            except Exception:
                idx = 0
            nxt = min(len(self.thumbnails) - 1, idx + 1)
            if nxt != idx:
                self._set_selected_thumb(self.thumbnails[nxt])
                # scroll canvas to make visible
                try:
                    widget = self.thumbnails[nxt]
                    self.canvas_gallery.xview_moveto(max(0, widget.winfo_x() / max(1, self.frame_thumbs.winfo_width())))
                except Exception:
                    pass
        except Exception:
            pass

    def _select_prev_thumb(self, event=None):
        try:
            if not self.thumbnails:
                return
            cur = getattr(self, '_selected_thumb', None)
            if cur is None:
                self._set_selected_thumb(self.thumbnails[0])
                return
            try:
                idx = self.thumbnails.index(cur)
            except Exception:
                idx = 0
            prv = max(0, idx - 1)
            if prv != idx:
                self._set_selected_thumb(self.thumbnails[prv])
                try:
                    widget = self.thumbnails[prv]
                    self.canvas_gallery.xview_moveto(max(0, widget.winfo_x() / max(1, self.frame_thumbs.winfo_width())))
                except Exception:
                    pass
        except Exception:
            pass

    def _on_thumb_motion(self, event, lbl):
        try:
            if not hasattr(self, '_drag_data') or self._drag_data.get('widget') is None:
                return
            if self._drag_ghost is None:
                # create a ghost window showing the image
                try:
                    self._drag_ghost = tk.Toplevel(self.root)
                    self._drag_ghost.overrideredirect(True)
                    # Robustly find a PhotoImage to show in the ghost.
                    img = None
                    # 1) direct on container
                    try:
                        img = getattr(lbl, 'image', None)
                    except Exception:
                        img = None
                    # 2) label child saved as image_label
                    if img is None:
                        try:
                            img = getattr(lbl, 'image_label', None) and getattr(lbl.image_label, 'image', None)
                        except Exception:
                            img = None
                    # 3) scan children for a label with .image attribute
                    if img is None:
                        try:
                            for child in lbl.winfo_children():
                                try:
                                    img = getattr(child, 'image', None)
                                    if img is not None:
                                        break
                                except Exception:
                                    continue
                        except Exception:
                            img = None
                    # 4) fallback: maybe the parameter was the image label itself
                    if img is None:
                        try:
                            img = getattr(self, 'last_known_thumb_image', None)
                        except Exception:
                            img = None
                    g_lbl = ttk.Label(self._drag_ghost, image=img)
                    g_lbl.image = img
                    g_lbl.pack()
                except Exception:
                    self._drag_ghost = None
                    return
            try:
                self._drag_ghost.geometry(f"+{event.x_root+8}+{event.y_root+8}")
            except Exception:
                pass
        except Exception:
            pass

    def _on_thumb_release(self, event, lbl):
        try:
            if getattr(self, '_drag_ghost', None) is not None:
                try:
                    self._drag_ghost.destroy()
                except Exception:
                    pass
                self._drag_ghost = None
            # compute new index based on event.x_root
            children = list(self.frame_thumbs.winfo_children())
            positions = []
            for c in children:
                try:
                    cx = c.winfo_rootx() + c.winfo_width() / 2
                    positions.append((c, cx))
                except Exception:
                    positions.append((c, 0))
            insert_idx = len(children)
            for i, (c, cx) in enumerate(positions):
                if event.x_root < cx:
                    insert_idx = i
                    break
            try:
                old_idx = self.thumbnails.index(lbl)
            except Exception:
                old_idx = None
            if old_idx is None:
                return
            # if user didn't move the thumbnail (click without drag), don't treat as reorder
            # compute adjusted insert index as the code would do when inserting
            adj_insert_idx = insert_idx
            if insert_idx > old_idx:
                adj_insert_idx = insert_idx - 1
            if adj_insert_idx == old_idx:
                # no movement; simply cleanup and return (don't trigger rename dialog)
                try:
                    # also set selection on click without move
                    try:
                        prev = getattr(self, '_selected_thumb', None)
                        if prev is not lbl:
                            try:
                                # stop pulse on previous
                                if prev is not None:
                                    try:
                                        self._stop_selection_pulse(prev)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                            self._set_selected_thumb(lbl)
                            try:
                                self._start_selection_pulse(lbl)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    del self._drag_data
                except Exception:
                    pass
                return
            # remove and insert (actual move)
            try:
                self.thumbnails.pop(old_idx)
                if insert_idx > old_idx:
                    insert_idx -= 1
                self.thumbnails.insert(insert_idx, lbl)
                self._repack_thumbnails()
                # after reordering, save order
                try:
                    self._save_folder_metadata()
                    if self.rename_on_reorder_var.get():
                        self._rename_files_by_order()
                except Exception:
                    pass
            except Exception:
                pass
            except Exception:
                pass
            finally:
                try:
                    del self._drag_data
                except Exception:
                    pass
        except Exception:
            pass

    # ---------------- curvature estimation / unwarp -----------------------
    def _estimate_spine_x(self, gray):
        """Estimate spine x position by finding the darkest vertical ridge in the central area."""
        try:
            h, w = gray.shape[:2]
            cx = w // 2
            # take vertical strip around center
            strip = gray[:, max(0, cx - w//6): min(w, cx + w//6)]
            # vertical projection (sum of dark pixels)
            v = np.mean(strip, axis=0)
            # find minima in projection (dark ridge)
            min_idx = int(np.argmin(v))
            spine_x = max(0, cx - w//6 + min_idx)
            return spine_x
        except Exception:
            return gray.shape[1] // 2

    def _unwarp_curvature(self, image):
        """Simple horizontal unwarp using estimated spine: remap x coordinates to flatten slight curvature.
        This is a lightweight approximation (cylindrical-like).
        """
        try:
            intensity = float(getattr(self, 'curvature_intensity_var', tk.DoubleVar(value=0.8)).get())
            cols = int(getattr(self, 'curvature_mesh_cols_var', tk.IntVar(value=40)).get())
            # use advanced mesh remap for stronger correction
            return mesh_unwarp_advanced(image, cols=cols, intensity=intensity)
        except Exception:
            try:
                return mesh_unwarp_advanced(image, cols=40, intensity=0.8)
            except Exception:
                return image

    def _open_preview(self, ruta):
        try:
            # if preview already shows this path, close it (toggle)
            try:
                if getattr(self, '_preview_path', None) == os.path.abspath(ruta):
                    return self._close_preview()
            except Exception:
                pass

            pil = Image.open(ruta)
            w, h = pil.size
            maxw, maxh = 1000, 900
            if w > maxw or h > maxh:
                pil.thumbnail((maxw, maxh), Image.LANCZOS)
            # reuse existing preview window if present
            try:
                if getattr(self, '_preview_top', None) and getattr(self, '_preview_top', 'destroyed') != 'destroyed':
                    top = self._preview_top
                    top.title(os.path.basename(ruta))
                    # replace image
                    img_tk = ImageTk.PhotoImage(pil)
                    if getattr(self, '_preview_label', None) is None:
                        lbl = ttk.Label(top, image=img_tk)
                        lbl.image = img_tk
                        lbl.pack(expand=True, fill='both')
                        self._preview_label = lbl
                    else:
                        lbl = self._preview_label
                        lbl.configure(image=img_tk)
                        lbl.image = img_tk
                    self._preview_path = os.path.abspath(ruta)
                    return
            except Exception:
                # fall through to create new
                pass

            top = tk.Toplevel(self.root)
            top.title(os.path.basename(ruta))
            img_tk = ImageTk.PhotoImage(pil)
            lbl = ttk.Label(top, image=img_tk)
            lbl.image = img_tk
            lbl.pack(expand=True, fill='both')
            btn = ttk.Button(top, text='Cerrar', command=lambda: self._close_preview())
            btn.pack(pady=6)
            self._preview_top = top
            self._preview_label = lbl
            self._preview_path = os.path.abspath(ruta)
        except Exception as e:
            print('Error mostrando preview:', e)

    def _close_preview(self):
        try:
            top = getattr(self, '_preview_top', None)
            if top:
                try:
                    top.destroy()
                except Exception:
                    pass
                self._preview_top = None
                self._preview_label = None
        except Exception:
            pass

    def _show_thumb_menu(self, event, lbl):
        try:
            menu = tk.Menu(self.root, tearoff=0)
            menu.add_command(label='Eliminar página', command=lambda l=lbl: self._delete_thumb(l))
            menu.add_command(label='Mover a la izquierda', command=lambda l=lbl: self._move_thumb(l, -1))
            menu.add_command(label='Mover a la derecha', command=lambda l=lbl: self._move_thumb(l, 1))
            menu.tk_popup(event.x_root, event.y_root)
        except Exception:
            pass

    def _repack_thumbnails(self):
        try:
            for child in list(self.frame_thumbs.winfo_children()):
                child.pack_forget()
            for lbl in self.thumbnails:
                lbl.pack(side='left', padx=5, pady=5)
        except Exception:
            pass

    def _move_thumb(self, lbl, direction):
        try:
            idx = self.thumbnails.index(lbl)
            new_idx = idx + direction
            if new_idx < 0 or new_idx >= len(self.thumbnails):
                return
            self.thumbnails[idx], self.thumbnails[new_idx] = self.thumbnails[new_idx], self.thumbnails[idx]
            self._repack_thumbnails()
            try:
                self._save_folder_metadata()
                if self.rename_on_reorder_var.get():
                    self._rename_files_by_order()
            except Exception:
                pass
        except Exception:
            pass

    def _delete_thumb(self, lbl):
        try:
            # confirm
            if not messagebox.askyesno('Confirmar', '¿Eliminar esta página? Esta acción no se puede deshacer.'):
                return
            ruta = getattr(lbl, 'filepath', None)
            if ruta and os.path.exists(ruta):
                try:
                    os.remove(ruta)
                except Exception:
                    pass
            try:
                lbl.destroy()
            except Exception:
                pass
            try:
                self.thumbnails.remove(lbl)
            except Exception:
                pass
            self._repack_thumbnails()
            try:
                self._save_folder_metadata()
            except Exception:
                pass
        except Exception:
            pass

    def _on_metadata_changed(self, *args):
        # called by StringVar trace; autosave if enabled and folder exists
        try:
            if getattr(self, 'autosave_var', None) and self.autosave_var.get():
                # small debounce: schedule save after short delay on main thread
                try:
                    self.root.after_cancel(getattr(self, '_autosave_after_id', None))
                except Exception:
                    pass
                try:
                    aid = self.root.after(300, lambda: self._save_folder_metadata())
                    self._autosave_after_id = aid
                except Exception:
                    # fallback immediate
                    self._save_folder_metadata()
        except Exception:
            pass

    # ---------------- camera -------------------------------------------------
    def _enumerar_camaras(self, max_test=6):
        disponibles = []
        backend_candidates = []
        if hasattr(cv2, 'CAP_DSHOW'):
            backend_candidates.append(cv2.CAP_DSHOW)
        if hasattr(cv2, 'CAP_MSMF'):
            backend_candidates.append(cv2.CAP_MSMF)
        backend_candidates.append(None)

        cam_entries = []
        for i in range(max_test + 1):
            found = False
            for backend in backend_candidates:
                cap = None
                try:
                    if backend is None:
                        cap = cv2.VideoCapture(i)
                    else:
                        cap = cv2.VideoCapture(i, backend)
                    if not cap.isOpened():
                        try:
                            cap.release()
                        except Exception:
                            pass
                        continue
                    ret, _ = cap.read()
                    try:
                        cap.release()
                    except Exception:
                        pass
                    if ret:
                        name = f"Cam {i}"
                        cam_entries.append((str(i), name))
                        disponibles.append(str(i))
                        found = True
                        break
                except Exception:
                    try:
                        if cap is not None:
                            cap.release()
                    except Exception:
                        pass
                    continue
            if found:
                continue

        if not disponibles:
            messagebox.showwarning("Cámara no encontrada", "No se ha detectado ninguna cámara disponible. Conecte una cámara y reinicie la aplicación.")
            disponibles = ['0']
            cam_entries = [("0", "0 - Desconocida")]

        values = [f"{idx} - {name}" for idx, name in cam_entries]
        if not values:
            values = disponibles
        self.cam_selector['values'] = values
        first = values[0]
        self.cam_selector.set(first)
        self.cam_var.set(first.split(' - ')[0])

    def _abrir_camara(self, index=0):
        try:
            if self.video is not None:
                try:
                    self.video.release()
                except Exception:
                    pass
        except Exception:
            pass

        idx = int(index)
        backend_candidates = []
        if hasattr(cv2, 'CAP_DSHOW'):
            backend_candidates.append(cv2.CAP_DSHOW)
        if hasattr(cv2, 'CAP_MSMF'):
            backend_candidates.append(cv2.CAP_MSMF)
        backend_candidates.append(None)

        cap = None
        for backend in backend_candidates:
            try:
                if backend is None:
                    cap_try = cv2.VideoCapture(idx)
                else:
                    cap_try = cv2.VideoCapture(idx, backend)
                if not cap_try.isOpened():
                    try:
                        cap_try.release()
                    except Exception:
                        pass
                    continue
                ret, frame = cap_try.read()
                if not ret or frame is None:
                    try:
                        cap_try.release()
                    except Exception:
                        pass
                    continue
                cap = cap_try
                break
            except Exception:
                try:
                    cap_try.release()
                except Exception:
                    pass
                continue

        if cap is None:
            messagebox.showerror("Error cámara", f"No se pudo abrir la cámara con índice {index}. Seleccione otro índice o conecte una cámara.")
            self.video = None
            self.cam_index = idx
            return

        # Try to set the camera to the highest common resolution that it supports
        common_res = [ (3840,2160), (2560,1440), (1920,1080), (1600,1200), (1280,720), (1024,768), (800,600) ]
        best = None
        for w,h in common_res:
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                time.sleep(0.1)
                ret, frame = cap.read()
                if ret and frame is not None and frame.shape[1] >= w-10 and frame.shape[0] >= h-10:
                    best = (w,h)
                    break
            except Exception:
                continue
        if best is not None:
            print(f"Usando resolución: {best[0]}x{best[1]} para cámara {idx}")

        self.video = cap
        self.cam_index = idx

    def _start_capture_thread(self):
        if self._capture_thread and self._capture_thread.is_alive():
            return
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def _capture_loop(self):
        while self._running:
            try:
                if self.video is None:
                    time.sleep(0.05)
                    continue
                ret, frame = self.video.read()
                if not ret or frame is None:
                    time.sleep(0.02)
                    continue
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                with self.frame_lock:
                    self.frame_actual = frame.copy()
            except Exception:
                time.sleep(0.05)

    def reiniciar_camara(self):
        sel = self.cam_selector.get() if self.cam_selector.get() else self.cam_var.get()
        idx = 0
        try:
            if isinstance(sel, str) and ' - ' in sel:
                idx = int(sel.split(' - ')[0])
            else:
                idx = int(sel)
        except Exception:
            idx = 0
        self._abrir_camara(idx)

    def cambiar_camara(self):
        self.reiniciar_camara()

    # ---------------- UI refresh / preview ----------------------------------
    def actualizar_video(self):
        frame = None
        with self.frame_lock:
            if self.frame_actual is not None:
                frame = self.frame_actual.copy()

        if frame is not None:
            try:
                display = frame.copy()
                try:
                    aw = getattr(self, 'aspect_ratio_weight', None) and float(self.aspect_ratio_weight.get()) or 1.0
                    ma = getattr(self, 'min_area_var', None) and int(self.min_area_var.get()) or 10000
                    use_bf = getattr(self, 'use_bright_fallback_var', None) and bool(self.use_bright_fallback_var.get())
                    debug_on = getattr(self, 'debug_det_var', None) and bool(self.debug_det_var.get())
                    topn = getattr(self, 'debug_topn_var', None) and int(self.debug_topn_var.get()) or 4
                except Exception:
                    aw = 1.0
                    ma = 10000
                    use_bf = True
                    debug_on = False
                    topn = 4

                det_res = detectar_libro(frame, min_area=ma, aspect_weight=aw, use_bright_fallback=use_bf, debug=debug_on)
                puntos = None
                candidates_dbg = None
                if debug_on and isinstance(det_res, tuple):
                    try:
                        puntos, candidates_dbg = det_res
                    except Exception:
                        puntos = det_res
                else:
                    puntos = det_res

                if puntos is not None:
                    try:
                        cv2.polylines(display, [np.int32(puntos)], True, (0, 255, 0), 3)
                    except Exception:
                        pass
                # draw debug overlays for top candidates (if provided)
                if debug_on and candidates_dbg:
                    try:
                        for i, c in enumerate(candidates_dbg[:topn]):
                            pts_c = np.array(c.get('pts', []), dtype=np.int32)
                            scr = c.get('score', 0.0)
                            if pts_c.size:
                                try:
                                    cv2.polylines(display, [pts_c], True, (0, 180, 255), 2)
                                    # draw score text near first point
                                    x, y = int(pts_c[0][0]), int(pts_c[0][1])
                                    cv2.putText(display, f"{scr:.1f}", (x+4, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                                except Exception:
                                    pass
                    except Exception:
                        pass

                preview = cv2.resize(display, (640, 480))
                if len(preview.shape) == 2:
                    preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
                if preview.shape[2] == 4:
                    preview = cv2.cvtColor(preview, cv2.COLOR_BGRA2BGR)
                preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                imgtk = ImageTk.PhotoImage(Image.fromarray(preview_rgb))
                self.lbl_video.imgtk = imgtk
                self.lbl_video.configure(image=imgtk)
                # Update status bar with camera and frame stats
                try:
                    cam_name = self.cam_selector.get() if self.cam_selector.get() else str(self.cam_index)
                    h, w = frame.shape[:2]
                    vmin = int(frame.min())
                    vmax = int(frame.max())
                    self.status_var.set(f"Cámara: {cam_name} | Resolución: {w}x{h} | min: {vmin} max: {vmax}")
                except Exception:
                    pass
            except Exception as e:
                print('Error al actualizar vista:', e)

        try:
            self.root.after(30, self.actualizar_video)
        except Exception:
            pass

    # ---------------- scan / save -------------------------------------------
    def escanear(self):
        # Check folder first to avoid showing modal dialogs while holding frame_lock
        if not self.carpeta_salida:
            messagebox.showwarning("Atención", "Primero cree la carpeta de salida y espere la vista previa.")
            return

        with self.frame_lock:
            if self.frame_actual is None:
                messagebox.showwarning("Atención", "Espere a que aparezca la vista previa antes de escanear.")
                return
            imagen = self.frame_actual.copy()

        # Start scan in background to avoid freezing UI
        if self._scanning:
            return

        self._scanning = True
        self._set_ui_enabled(False)
        self.status_var.set("Escaneando...")
        thread = threading.Thread(target=self._do_scan, args=(imagen,), daemon=True)
        thread.start()

    def _set_ui_enabled(self, enabled: bool):
        try:
            state = 'normal' if enabled else 'disabled'
            for btn in (getattr(self, 'btn_crear', None), getattr(self, 'btn_capturar', None), getattr(self, 'btn_export', None), getattr(self, 'btn_reiniciar', None), getattr(self, 'btn_salir', None)):
                if btn is None:
                    continue
                try:
                    btn.configure(state=state)
                except Exception:
                    pass
            try:
                if enabled:
                    self.cam_selector.configure(state='readonly')
                else:
                    self.cam_selector.configure(state='disabled')
            except Exception:
                pass
        except Exception:
            pass

    def exportar_pdf(self):
        if not self.carpeta_salida:
            messagebox.showwarning("Atención", "Cree primero la carpeta de salida.")
            return
        archivos = sorted([f for f in os.listdir(self.carpeta_salida) if f.lower().endswith('.png')])
        if not archivos:
            messagebox.showinfo("Sin páginas", "No hay imágenes para exportar.")
            return
        imagenes = [Image.open(os.path.join(self.carpeta_salida, f)).convert("RGB") for f in archivos]
        pdf_path = os.path.join(self.carpeta_salida, f"{self.titulo_var.get().strip() or 'documento'}.pdf")
        imagenes[0].save(pdf_path, save_all=True, append_images=imagenes[1:])
        # add metadata
        try:
            self.agregar_metadatos_pdf(pdf_path)
        except Exception:
            pass
        # autosave folder metadata after export
        try:
            if getattr(self, 'autosave_var', None) and self.autosave_var.get():
                self._save_folder_metadata()
        except Exception:
            pass
        messagebox.showinfo("Exportación completada", f"PDF generado:\n{pdf_path}")

    def _do_scan(self, imagen, timeout=8.0):
        start = time.time()
        pts = None
        try:
            # Try primary detector, but guard with timeout
            while True:
                if time.time() - start > timeout:
                    break
                # pass UI-tunable params if available
                try:
                    min_area = int(getattr(self, 'min_area_var', tk.IntVar(value=10000)).get())
                except Exception:
                    min_area = 10000
                try:
                    aw = getattr(self, 'aspect_ratio_weight', None) and float(self.aspect_ratio_weight.get()) or 1.0
                    use_bf = getattr(self, 'use_bright_fallback_var', None) and bool(self.use_bright_fallback_var.get())
                    debug_on = getattr(self, 'debug_det_var', None) and bool(self.debug_det_var.get())
                except Exception:
                    aw = 1.0
                    use_bf = True
                    debug_on = False
                det_res = detectar_libro(imagen, min_area=min_area, aspect_weight=aw, use_bright_fallback=use_bf, debug=debug_on)
                if debug_on and isinstance(det_res, tuple):
                    try:
                        pts, dbg = det_res
                    except Exception:
                        pts = det_res
                else:
                    pts = det_res
                if pts is not None:
                    break
                # try a quicker adaptive-threshold fallback
                gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(c) > 5000:
                        peri = cv2.arcLength(c, True)
                        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                        if len(approx) >= 4:
                            pts = approx.reshape(-1, 2)[:4]
                            break
                time.sleep(0.05)

            if pts is not None:
                try:
                    imagen_proc = four_point_transform(imagen, pts)
                    # optional curvature correction
                    try:
                        if getattr(self, 'curvature_var', None) and self.curvature_var.get():
                            imagen_proc = self._unwarp_curvature(imagen_proc)
                    except Exception:
                        pass
                except Exception:
                    imagen_proc = imagen
                # if debug was requested, save overlay + candidates info for this scan
                try:
                    if debug_on:
                        try:
                            outdir = os.path.join(os.getcwd(), 'output', 'temp_test')
                            os.makedirs(outdir, exist_ok=True)
                            # build overlay from original display-size image
                            overlay = imagen.copy()
                            # draw detected polygon
                            try:
                                cv2.polylines(overlay, [np.int32(pts)], True, (0, 255, 0), 3)
                            except Exception:
                                pass
                            # draw candidate polygons if dbg present
                            try:
                                if 'dbg' in locals() and isinstance(dbg, list):
                                    for i, c in enumerate(dbg):
                                        try:
                                            pts_c = np.array(c.get('pts', []), dtype=np.int32)
                                            score = c.get('score', 0.0)
                                            if pts_c.size:
                                                cv2.polylines(overlay, [pts_c], True, (0, 180, 255), 2)
                                                x, y = int(pts_c[0][0]), int(pts_c[0][1])
                                                cv2.putText(overlay, f"{score:.1f}", (x+4, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                            debug_img_path = os.path.join(outdir, f"scan_debug_{ts}.jpg")
                            cv2.imwrite(debug_img_path, overlay)
                            # save candidates JSON
                            try:
                                dbg_info = {'detected_pts': pts.tolist() if hasattr(pts, 'tolist') else None, 'candidates': dbg if isinstance(dbg, list) else None, 'timestamp': ts}
                                with open(os.path.join(outdir, f"scan_debug_{ts}.json"), 'w', encoding='utf-8') as jf:
                                    json.dump(dbg_info, jf, ensure_ascii=False, indent=2)
                            except Exception:
                                pass
                        except Exception:
                            pass
                except Exception:
                    pass
            else:
                h, w = imagen.shape[:2]
                pad_w = int(w * 0.05)
                pad_h = int(h * 0.05)
                imagen_proc = imagen[pad_h:h - pad_h, pad_w:w - pad_w]

            if self.split_var.get():
                try:
                    izq, der = separar_paginas(imagen_proc)
                    paginas = [izq, der]
                except Exception:
                    paginas = [imagen_proc]
            else:
                paginas = [imagen_proc]

            saved = []
            for i, pag in enumerate(paginas, start=1):
                # build safe filename: {signatura}{title}_{contador:03d}_p{i}.png
                sign = self.signatura_var.get().strip() or ''
                title = self.titulo_var.get().strip() or 'escaneo'
                def_suf = f"_p{i}"
                # sanitize components
                def _sanitize(s):
                    return re.sub(r"[^A-Za-z0-9\-_ ]+", '', s).strip().replace(' ', '_')
                sign_s = _sanitize(sign)
                title_s = _sanitize(title)
                nombre = f"{sign_s}{title_s}_{self.contador:03d}{def_suf}.png"
                ruta = os.path.join(self.carpeta_salida, nombre)
                try:
                    if len(pag.shape) == 2:
                        pag_save = cv2.cvtColor(pag, cv2.COLOR_GRAY2BGR)
                    elif pag.shape[2] == 4:
                        pag_save = cv2.cvtColor(pag, cv2.COLOR_BGRA2BGR)
                    else:
                        pag_save = pag
                    cv2.imwrite(ruta, pag_save)
                    saved.append((ruta, pag_save))
                except Exception as e:
                    print('Error guardando página:', e)

            def add_thumbs():
                for ruta, pag_save in saved:
                    try:
                        thumb = cv2.resize(pag_save, (int(self.thumb_w_var.get()), int(self.thumb_h_var.get())))
                        thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(thumb)
                        self._add_thumbnail_from_pil(pil_img, ruta)
                        print(f"Guardado: {ruta}")
                    except Exception as e:
                        print('Error creando miniatura:', e)

            try:
                self.root.after(10, add_thumbs)
            except Exception:
                pass

            self.contador += 1
        finally:
            self._scanning = False
            # autosave metadata after scan (update counter etc.)
            try:
                if getattr(self, 'autosave_var', None) and self.autosave_var.get():
                    self._save_folder_metadata()
            except Exception:
                pass
            try:
                self.root.after(50, lambda: (self._set_ui_enabled(True), self.status_var.set('Listo')))
            except Exception:
                self._set_ui_enabled(True)

    def agregar_metadatos_pdf(self, pdf_path):
        try:
            reader = PdfReader(pdf_path)
            writer = PdfWriter()
            for p in reader.pages:
                writer.add_page(p)
            tema = self.tema_var.get().strip()
            now = datetime.datetime.now().isoformat()
            metadata = {
                "/Title": self.titulo_var.get().strip() or "",
                "/Author": self.autor_var.get().strip() or "",
                "/Subject": tema or "",
                "/CreationDate": now,
                "/ModDate": now,
                "/Keywords": tema or "",
            }
            # PyPDF2 expects keys starting with '/', writer.add_metadata
            writer.add_metadata(metadata)
            temp_pdf = pdf_path + ".tmp"
            with open(temp_pdf, "wb") as f:
                writer.write(f)
            os.replace(temp_pdf, pdf_path)
        except Exception as e:
            print('Error agregando metadatos PDF:', e)

    def crear_carpeta(self):
        signatura = self.signatura_var.get().strip()
        if not signatura:
            messagebox.showwarning("Atención", "Debe ingresar una signatura para crear la carpeta.")
            return
        self.carpeta_salida = os.path.join(os.getcwd(), signatura)
        os.makedirs(self.carpeta_salida, exist_ok=True)
        messagebox.showinfo("Carpeta creada", f"Las imágenes se guardarán en:\n{self.carpeta_salida}")
        # save folder metadata and last session
        try:
            # cleanup any leftover tmp files from previous interrupted operations
            try:
                self._cleanup_tmp_files()
            except Exception:
                pass
            self._save_folder_metadata()
            self._save_last_session()
            # load any existing thumbnails (in case folder already had images)
            try:
                self._load_existing_thumbnails()
            except Exception:
                pass
        except Exception:
            pass

    def salir(self):
        self._running = False
        try:
            if self.video is not None:
                self.video.release()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    # ---------------- folder / metadata persistence -----------------------
    def abrir_carpeta(self):
        path = filedialog.askdirectory(title="Abrir carpeta de trabajo")
        if not path:
            return
        # load metadata if exists
        self.carpeta_salida = path
        try:
            # cleanup any leftover tmp files
            try:
                self._cleanup_tmp_files()
            except Exception:
                pass
            self._load_folder_metadata()
            # populate gallery with existing images
            try:
                self._load_existing_thumbnails()
            except Exception:
                pass
            self._save_last_session()
            messagebox.showinfo("Carpeta abierta", f"Carpeta abierta:\n{self.carpeta_salida}")
        except Exception as e:
            print('Error al abrir carpeta:', e)

    def _folder_metadata_path(self):
        if not self.carpeta_salida:
            return None
        return os.path.join(self.carpeta_salida, 'metadata.json')

    def _save_folder_metadata(self):
        path = self._folder_metadata_path()
        if not path:
            return
        data = {
            'titulo': self.titulo_var.get().strip(),
            'autor': self.autor_var.get().strip(),
            'tema': self.tema_var.get().strip(),
            'signatura': self.signatura_var.get().strip(),
            'archivo': self.archivo_var.get().strip(),
            'contador': self.contador,
        }
        # persist explicit order of thumbnails (filenames)
        try:
            order = [os.path.basename(getattr(lbl, 'filepath', '')) for lbl in self.thumbnails if getattr(lbl, 'filepath', None)]
            if order:
                data['order'] = order
        except Exception:
            pass
        # persist timestamp
        try:
            now = datetime.datetime.now().isoformat()
            data['last_saved'] = now
        except Exception:
            data['last_saved'] = None
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            # update UI label
            try:
                if data.get('last_saved'):
                    ts = datetime.datetime.fromisoformat(data['last_saved']).strftime('%H:%M:%S')
                    self._last_saved_var.set(f"Metadatos guardados: {ts}")
                else:
                    self._last_saved_var.set("Metadatos guardados: -")
            except Exception:
                pass
        except Exception as e:
            print('Error guardando metadata folder:', e)

        # persist selected thumbnail basename if present
        try:
            sel = getattr(self, '_selected_thumb', None)
            if sel and getattr(sel, 'filepath', None):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        d = json.load(f)
                except Exception:
                    d = data
                try:
                    d['selected'] = os.path.basename(sel.filepath)
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(d, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
        except Exception:
            pass

    def _apply_desired_order(self):
        """Reorder self.thumbnails widgets according to the filenames in _desired_thumb_order."""
        try:
            desired = getattr(self, '_desired_thumb_order', None)
            if not desired:
                return
            # build a map from basename -> widget
            mapping = {}
            for lbl in list(self.thumbnails):
                fp = getattr(lbl, 'filepath', None)
                if not fp:
                    continue
                mapping[os.path.basename(fp)] = lbl
            new_list = []
            for name in desired:
                w = mapping.get(os.path.basename(name))
                if w:
                    new_list.append(w)
            # append any remaining widgets not in desired order
            for lbl in self.thumbnails:
                if lbl not in new_list:
                    new_list.append(lbl)
            self.thumbnails = new_list
            self._repack_thumbnails()
        except Exception:
            pass

    def _rename_files_by_order(self):
        """Safely rename image files to reflect the current thumbnail order.

        This performs a two-phase rename using temporary names to avoid collisions.
        Shows a confirmation dialog before performing destructive renaming.
        """
        try:
            if not self.carpeta_salida:
                return
            if not self.thumbnails:
                return
            # Confirm
            if not messagebox.askyesno('Confirmar renombrado', 'Renombrar los archivos para reflejar el nuevo orden? Esta acción modifica los nombres de archivo.'):
                return
            # Build new names based on title and index
            sign = self.signatura_var.get().strip() or ''
            title = self.titulo_var.get().strip() or 'escaneo'
            # sanitize
            def _sanitize(s):
                return re.sub(r"[^A-Za-z0-9\-_ ]+", '', s).strip().replace(' ', '_')
            sign_s = _sanitize(sign)
            title_s = _sanitize(title)
            new_names = []
            for i, lbl in enumerate(self.thumbnails, start=1):
                # try to preserve page suffix from original filename (_p1/_p2) if present
                orig = os.path.basename(getattr(lbl, 'filepath', ''))
                m = re.search(r"(_p\d+)(\.[^.]+)?$", orig)
                page_suffix = m.group(1) if m else f"_p1"
                ext = os.path.splitext(orig)[1] or '.png'
                new_names.append(f"{sign_s}{title_s}_{i:03d}{page_suffix}{ext}")
            # perform two-phase rename: move to .tmp names first
            tmp_names = []
            # backup mapping old->new for undo
            backup = []
            for lbl, new in zip(self.thumbnails, new_names):
                old = getattr(lbl, 'filepath', None)
                if not old or not os.path.exists(old):
                    tmp_names.append((None, None))
                    backup.append((old, None))
                    continue
                tmp = os.path.join(self.carpeta_salida, new + '.tmp')
                final = os.path.join(self.carpeta_salida, new)
                try:
                    os.replace(old, tmp)
                    tmp_names.append((tmp, final))
                    backup.append((final, old))
                except Exception:
                    # abort: try to rollback any moved files
                    for moved_tmp, moved_final in tmp_names:
                        try:
                            if moved_tmp and os.path.exists(moved_tmp):
                                os.replace(moved_tmp, moved_final)
                        except Exception:
                            pass
                    messagebox.showerror('Error', 'No se pudo renombrar los archivos. Operación abortada.')
                    return
            # commit phase: rename tmp -> final
            for tmp, final in tmp_names:
                if not tmp:
                    continue
                try:
                    os.replace(tmp, final)
                except Exception:
                    # best-effort: continue
                    pass
            # persist backup mapping to metadata for undo
            try:
                mdpath = self._folder_metadata_path()
                if mdpath and os.path.exists(mdpath):
                    try:
                        with open(mdpath, 'r', encoding='utf-8') as mf:
                            mdata = json.load(mf)
                    except Exception:
                        mdata = {}
                else:
                    mdata = {}
                mdata['last_rename_backup'] = [{'new': n, 'old': o} for (n, o) in backup if n or o]
                try:
                    with open(mdpath, 'w', encoding='utf-8') as mf:
                        json.dump(mdata, mf, ensure_ascii=False, indent=2)
                except Exception:
                    pass
            except Exception:
                pass
            # update lbl.filepath for widgets
            for lbl, new in zip(self.thumbnails, new_names):
                try:
                    newpath = os.path.join(self.carpeta_salida, new)
                    lbl.filepath = newpath
                    # update visible filename label if present
                    try:
                        base = os.path.splitext(os.path.basename(newpath))[0]
                        if getattr(lbl, 'name_label', None):
                            lbl.name_label.configure(text=base)
                    except Exception:
                        pass
                except Exception:
                    pass
            # save updated metadata
            try:
                self._save_folder_metadata()
            except Exception:
                pass
            messagebox.showinfo('Renombrado', 'Renombrado completado.')
        except Exception as e:
            print('Error renombrando archivos por orden:', e)

    def _undo_last_rename(self):
        try:
            mdpath = self._folder_metadata_path()
            if not mdpath or not os.path.exists(mdpath):
                messagebox.showinfo('Deshacer', 'No hay respaldo para deshacer.')
                return
            with open(mdpath, 'r', encoding='utf-8') as mf:
                mdata = json.load(mf)
            backup = mdata.get('last_rename_backup')
            if not backup:
                messagebox.showinfo('Deshacer', 'No hay respaldo para deshacer.')
                return
            # ask confirm
            if not messagebox.askyesno('Deshacer renombrado', '¿Desea revertir el último renombrado?'):
                return
            # perform reverse mapping new->old
            for item in reversed(backup):
                new = item.get('new')
                old = item.get('old')
                if not new or not old:
                    continue
                new_path = os.path.join(self.carpeta_salida, os.path.basename(new))
                old_path = os.path.join(self.carpeta_salida, os.path.basename(old))
                try:
                    if os.path.exists(new_path):
                        # if old exists, create unique backup name
                        if os.path.exists(old_path):
                            try:
                                os.remove(old_path)
                            except Exception:
                                pass
                        os.replace(new_path, old_path)
                except Exception:
                    pass
            # clear backup entry
            try:
                mdata['last_rename_backup'] = []
                with open(mdpath, 'w', encoding='utf-8') as mf:
                    json.dump(mdata, mf, ensure_ascii=False, indent=2)
            except Exception:
                pass
            # refresh gallery
            try:
                self._load_existing_thumbnails()
            except Exception:
                pass
            messagebox.showinfo('Deshacer', 'Operación de deshacer completada.')
        except Exception as e:
            print('Error deshaciendo renombrado:', e)

    def _load_folder_metadata(self):
        path = self._folder_metadata_path()
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.titulo_var.set(data.get('titulo', ''))
            self.autor_var.set(data.get('autor', ''))
            self.tema_var.set(data.get('tema', ''))
            self.signatura_var.set(data.get('signatura', ''))
            self.archivo_var.set(data.get('archivo', ''))
            self.contador = data.get('contador', self.contador)
            # restore last_saved indicator
            try:
                ls = data.get('last_saved')
                if ls:
                    ts = datetime.datetime.fromisoformat(ls).strftime('%H:%M:%S')
                    self._last_saved_var.set(f"Metadatos guardados: {ts}")
                else:
                    self._last_saved_var.set("Metadatos guardados: -")
            except Exception:
                pass
            # once metadata loaded, populate gallery
            try:
                self._load_existing_thumbnails()
            except Exception:
                pass
            # after thumbnails loaded, restore selection if present
            try:
                sel = data.get('selected')
                if sel:
                    # wait briefly for thumbnails to load then select
                    def _restore():
                        try:
                            for lbl in self.thumbnails:
                                if os.path.basename(getattr(lbl, 'filepath', '')) == sel:
                                    try:
                                        self._set_selected_thumb(lbl)
                                        self._start_selection_pulse(lbl)
                                    except Exception:
                                        pass
                                    break
                        except Exception:
                            pass
                    try:
                        self.root.after(200, _restore)
                    except Exception:
                        _restore()
            except Exception:
                pass
        except Exception as e:
            print('Error cargando metadata folder:', e)

    def _last_session_path(self):
        # store a last-session file in the project folder
        return os.path.join(os.getcwd(), '.last_session.json')

    def _save_last_session(self):
        try:
            p = self._last_session_path()
            data = {
                'carpeta_salida': self.carpeta_salida,
                'cam_index': self.cam_index,
                'tmp_policy': getattr(self, 'tmp_policy_var', None) and self.tmp_policy_var.get() or 'ask',
                # detection UI settings
                'detection': {
                    'aspect_ratio_weight': getattr(self, 'aspect_ratio_weight', None) and float(self.aspect_ratio_weight.get()) or 1.0,
                    'use_bright_fallback': getattr(self, 'use_bright_fallback_var', None) and bool(self.use_bright_fallback_var.get()) or False,
                    'min_area': getattr(self, 'min_area_var', None) and int(self.min_area_var.get()) or 10000,
                    'debug_det': getattr(self, 'debug_det_var', None) and bool(self.debug_det_var.get()) or False
                }
            }
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print('Error guardando last session:', e)

    def _load_last_session(self):
        try:
            p = self._last_session_path()
            if not os.path.exists(p):
                return
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            last = data.get('carpeta_salida')
            if last and os.path.isdir(last):
                self.carpeta_salida = last
                # try loading folder metadata
                try:
                    self._load_folder_metadata()
                except Exception:
                    pass
                # also populate gallery from any existing images even if metadata missing
                try:
                    # cleanup any leftover tmp files before loading
                    try:
                        self._cleanup_tmp_files()
                    except Exception:
                        pass
                    self._load_existing_thumbnails()
                except Exception:
                    pass
            ci = data.get('cam_index')
            if isinstance(ci, int):
                self.cam_index = ci
            # restore tmp policy if present
            try:
                tp = data.get('tmp_policy')
                if tp and getattr(self, 'tmp_policy_var', None) is not None:
                    self.tmp_policy_var.set(tp)
                    # ensure combobox reflects value
                    try:
                        self.tmp_policy_combo.set(tp)
                    except Exception:
                        pass
            except Exception:
                pass
            # restore detection UI settings if present
            try:
                det = data.get('detection')
                if det:
                    try:
                        arw = det.get('aspect_ratio_weight')
                        if arw is not None and getattr(self, 'aspect_ratio_weight', None) is not None:
                            self.aspect_ratio_weight.set(float(arw))
                    except Exception:
                        pass
                    try:
                        ubf = det.get('use_bright_fallback')
                        if ubf is not None and getattr(self, 'use_bright_fallback_var', None) is not None:
                            self.use_bright_fallback_var.set(bool(ubf))
                    except Exception:
                        pass
                    try:
                        ma = det.get('min_area')
                        if ma is not None and getattr(self, 'min_area_var', None) is not None:
                            self.min_area_var.set(int(ma))
                    except Exception:
                        pass
                    try:
                        dbg = det.get('debug_det')
                        if dbg is not None and getattr(self, 'debug_det_var', None) is not None:
                            self.debug_det_var.set(bool(dbg))
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception as e:
            print('Error cargando last session:', e)


if __name__ == "__main__":
    root = tk.Tk()
    app = BookScannerApp(root)
    root.mainloop()
    
