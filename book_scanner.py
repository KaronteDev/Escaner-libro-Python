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


def detectar_libro(frame, canny1=75, canny2=200, min_area=10000):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny1, canny2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
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
    dst = np.array([[0, 0], [0, maxAlto - 1], [maxAncho - 1, maxAlto - 1], [maxAncho - 1, 0]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxAncho, maxAlto))


def separar_paginas(imagen):
    h, w = imagen.shape[:2]
    mitad = w // 2
    return imagen[:, :mitad], imagen[:, mitad:]


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
        frame_form = ttk.LabelFrame(self.root, text="Datos del documento", padding=10)
        frame_form.pack(side="left", fill="y", padx=10, pady=10)
        self.frame_form = frame_form

        ttk.Label(frame_form, text="Título del libro:").pack(anchor="w")
        ttk.Entry(frame_form, textvariable=self.titulo_var).pack(fill="x")

        ttk.Label(frame_form, text="Autor:").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame_form, textvariable=self.autor_var).pack(fill="x")

        ttk.Label(frame_form, text="Tema investigación:").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame_form, textvariable=self.tema_var).pack(fill="x")

        ttk.Label(frame_form, text="Signatura:").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame_form, textvariable=self.signatura_var).pack(fill="x")

        ttk.Label(frame_form, text="Archivo/Biblioteca:").pack(anchor="w", pady=(10, 0))

        ttk.Entry(frame_form, textvariable=self.archivo_var).pack(fill="x")
        
        # store buttons so we can enable/disable them during scanning
        self.btn_crear = ttk.Button(frame_form, text="📁 Crear carpeta", command=self.crear_carpeta)
        self.btn_crear.pack(fill="x", pady=(15, 0))
        # Open existing folder to continue working
        self.btn_abrir = ttk.Button(frame_form, text="📂 Abrir carpeta", command=self.abrir_carpeta)
        self.btn_abrir.pack(fill="x", pady=(8, 0))
        self.btn_export = ttk.Button(frame_form, text="🧾 Exportar a PDF", command=self.exportar_pdf)
        self.btn_export.pack(fill="x", pady=(10, 0))
        self.btn_reiniciar = ttk.Button(frame_form, text="🔁 Reiniciar cámara", command=self.reiniciar_camara)
        self.btn_reiniciar.pack(fill="x", pady=(10, 0))
        self.btn_salir = ttk.Button(frame_form, text="❌ Salir", command=self.salir)
        self.btn_salir.pack(fill="x", pady=(10, 0))
        
        
        # Auto-save metadata option
        self.autosave_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame_form, text="Auto-guardar metadatos", variable=self.autosave_var).pack(fill="x", pady=(6, 0))

        # Thumbnail size controls
        ttk.Label(frame_form, text="Tamaño miniatura (px):").pack(anchor="w", pady=(8, 0))
        self.thumb_w_var = tk.IntVar(value=120)
        self.thumb_h_var = tk.IntVar(value=160)
        size_frame = ttk.Frame(frame_form)
        size_frame.pack(fill="x")
        tk.Spinbox(size_frame, from_=60, to=400, textvariable=self.thumb_w_var, width=6).pack(side="left")
        ttk.Label(size_frame, text="x").pack(side="left", padx=4)
        tk.Spinbox(size_frame, from_=60, to=400, textvariable=self.thumb_h_var, width=6).pack(side="left")
        ttk.Button(size_frame, text="Aplicar", command=lambda: (self._load_existing_thumbnails())).pack(side="left", padx=8)
        # Optionally rename files to preserve order when reordering
        self.rename_on_reorder_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_form, text="Renombrar archivos al reordenar", variable=self.rename_on_reorder_var).pack(fill="x", pady=(6, 0))

        # Undo last rename (backup) button
        self.btn_undo_rename = ttk.Button(frame_form, text="↶ Deshacer renombrado", command=self._undo_last_rename)
        self.btn_undo_rename.pack(fill="x", pady=(6, 0))

        # .tmp policy control (ask / commit / delete)
        ttk.Label(frame_form, text="Política archivos .tmp:").pack(anchor='w', pady=(8,0))
        self.tmp_policy_var = tk.StringVar(value='ask')
        self.tmp_policy_combo = ttk.Combobox(frame_form, textvariable=self.tmp_policy_var, state='readonly', values=['ask','commit','delete'])
        self.tmp_policy_combo.pack(fill='x')

        # Last-saved indicator
        self._last_saved_var = tk.StringVar(value="Metadatos guardados: -")
        ttk.Label(frame_form, textvariable=self._last_saved_var, foreground="#2e7d32").pack(fill="x", pady=(4, 0))

      

        # Camera selector
        frame_cam = ttk.Frame(frame_form)
        frame_cam.pack(fill="x", pady=(10, 0))
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
            policy = getattr(self, 'tmp_policy_var', None) and self.tmp_policy_var.get() or 'ask'
            if policy == 'delete':
                for t in tmps:
                    try:
                        os.remove(os.path.join(self.carpeta_salida, t))
                    except Exception:
                        pass
                return
            elif policy == 'commit':
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
            container = ttk.Frame(self.frame_thumbs)
            lbl_img = ttk.Label(container, image=img_tk)
            lbl_img.image = img_tk
            lbl_img.pack()
            # filename label (no extension)
            fname = os.path.splitext(os.path.basename(ruta))[0]
            lbl_name = ttk.Label(container, text=fname, width=16, anchor='center')
            lbl_name.pack()
            # attach filepath on container for consistency
            container.filepath = ruta
            container.image_label = lbl_img
            container.name_label = lbl_name
            container.pack(side="left", padx=5, pady=5)
            # left click opens preview (bind on image label)
            lbl_img.bind('<Button-1>', lambda e, r=ruta: self._open_preview(r))
            # right click shows context menu (bind on container)
            container.bind('<Button-3>', lambda e, c=container: self._show_thumb_menu(e, c))
            # drag-and-drop bindings (bind on container)
            container.bind('<ButtonPress-1>', lambda e, c=container: self._on_thumb_press(e, c))
            container.bind('<B1-Motion>', lambda e, c=container: self._on_thumb_motion(e, c))
            container.bind('<ButtonRelease-1>', lambda e, c=container: self._on_thumb_release(e, c))
            self.thumbnails.append(container)
        except Exception:
            pass

    def _on_thumb_press(self, event, lbl):
        try:
            self._drag_data = {'widget': lbl, 'start_x': event.x_root, 'start_y': event.y_root}
            self._drag_ghost = None
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
                    img = lbl.image
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
            # remove and insert
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
            finally:
                try:
                    del self._drag_data
                except Exception:
                    pass
        except Exception:
            pass

    def _open_preview(self, ruta):
        try:
            pil = Image.open(ruta)
            w, h = pil.size
            maxw, maxh = 1000, 900
            if w > maxw or h > maxh:
                pil.thumbnail((maxw, maxh), Image.LANCZOS)
            top = tk.Toplevel(self.root)
            top.title(os.path.basename(ruta))
            img_tk = ImageTk.PhotoImage(pil)
            lbl = ttk.Label(top, image=img_tk)
            lbl.image = img_tk
            lbl.pack(expand=True, fill='both')
            btn = ttk.Button(top, text='Cerrar', command=top.destroy)
            btn.pack(pady=6)
        except Exception as e:
            print('Error mostrando preview:', e)

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
                puntos = detectar_libro(frame)
                if puntos is not None:
                    try:
                        cv2.polylines(display, [np.int32(puntos)], True, (0, 255, 0), 3)
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
                pts = detectar_libro(imagen)
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
                except Exception:
                    imagen_proc = imagen
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
                    lbl.filepath = os.path.join(self.carpeta_salida, new)
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
        except Exception as e:
            print('Error cargando last session:', e)


if __name__ == "__main__":
    root = tk.Tk()
    app = BookScannerApp(root)
    root.mainloop()
    
