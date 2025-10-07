import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import datetime

# -------------------------------
# Funciones de procesamiento
# -------------------------------

def ordenar_puntos(puntos):
    puntos = sorted(puntos, key=lambda x: x[0])
    izq = sorted(puntos[:2], key=lambda x: x[1])
    der = sorted(puntos[2:], key=lambda x: x[1])
    return np.array([izq[0], izq[1], der[1], der[0]], dtype="float32")

def detectar_libro(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contornos_ordenados = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contornos_ordenados:
        perimetro = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimetro, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None

def aplanar_imagen(frame, puntos):
    orden = ordenar_puntos(puntos)
    (tl, bl, br, tr) = orden
    ancho = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    alto = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    destino = np.array([[0, 0], [0, alto - 1], [ancho - 1, alto - 1], [ancho - 1, 0]], dtype="float32")
    matriz = cv2.getPerspectiveTransform(orden, destino)
    imagen_aplanada = cv2.warpPerspective(frame, matriz, (int(ancho), int(alto)))
    return imagen_aplanada

def dividir_paginas(imagen):
    h, w, _ = imagen.shape
    mitad = w // 2
    izq = imagen[:, :mitad]
    der = imagen[:, mitad:]
    return izq, der

# -------------------------------
# Clase GUI principal
# -------------------------------

class BookScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📘 Escáner de Libros - OpenCV")

        # Cámara / frames
        self.cap = None
        self.frame_actual = None
        self.contador = 1
        self.output_dir = ""

        # Flags para controlar el bucle de actualización y evitar reentradas
        self._updating = False
        self._running = True

        # ---- Formulario ----
        frm = ttk.LabelFrame(self.root, text="Metadatos del Documento", padding=10)
        frm.pack(fill="x", padx=10, pady=10)

        self.var_nombre = tk.StringVar()
        self.var_autor = tk.StringVar()
        self.var_signatura = tk.StringVar()
        self.var_archivo = tk.StringVar()

        ttk.Label(frm, text="Título del libro:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.var_nombre, width=50).grid(row=0, column=1)

        ttk.Label(frm, text="Autor:").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.var_autor, width=50).grid(row=1, column=1)

        ttk.Label(frm, text="Signatura (ID carpeta):").grid(row=2, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.var_signatura, width=50).grid(row=2, column=1)

        ttk.Label(frm, text="Archivo / Biblioteca:").grid(row=3, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.var_archivo, width=50).grid(row=3, column=1)

        ttk.Button(frm, text="Iniciar Escaneo", command=self.iniciar_camara).grid(row=4, column=1, pady=10)

        # ---- Área de video ----
        self.lbl_video = ttk.Label(self.root)
        self.lbl_video.pack()

        # Bindings
        self.root.bind("<space>", self.capturar_pagina)
        self.root.protocol("WM_DELETE_WINDOW", self.cerrar)

    def iniciar_camara(self):
        if not self.var_signatura.get().strip():
            messagebox.showwarning("Falta información", "Introduce una signatura antes de continuar.")
            return

        self.output_dir = os.path.join("output", self.var_signatura.get().strip())
        os.makedirs(self.output_dir, exist_ok=True)

        # Abrir cámara por defecto
        self.cap = cv2.VideoCapture(0)
        # Asegurar que la bandera de ejecución está activa
        self._running = True
        self.actualizar_video()

    def actualizar_video(self):
        # Evitar reentrada si la actualización anterior sigue en curso
        if not getattr(self, '_running', True):
            return
        if getattr(self, '_updating', False):
            # reprogramar y salir
            try:
                self.root.after(10, self.actualizar_video)
            except Exception:
                pass
            return

        self._updating = True
        try:
            if self.cap is None:
                return
            ret, frame = self.cap.read()
            if ret and frame is not None:
                frame = cv2.flip(frame, 1)
                # Normalizar número de canales
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # Detección de contorno (si se detecta, dibujar)
                puntos = detectar_libro(frame)
                display = frame.copy()
                if puntos is not None:
                    try:
                        cv2.polylines(display, [np.int32(puntos)], True, (0, 255, 0), 3)
                    except Exception:
                        pass

                # Redimensionar para la vista previa y convertir a RGB
                preview = cv2.resize(display, (640, 480))
                if len(preview.shape) == 2:
                    preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
                if preview.shape[2] == 4:
                    preview = cv2.cvtColor(preview, cv2.COLOR_BGRA2BGR)
                preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                imgtk = ImageTk.PhotoImage(Image.fromarray(preview_rgb))
                try:
                    self.lbl_video.imgtk = imgtk
                    self.lbl_video.configure(image=imgtk)
                except Exception as e:
                    print('Error updating GUI image:', e)
        except Exception as e:
            print('Exception in actualizar_video:', e)
        finally:
            self._updating = False
            if getattr(self, '_running', True):
                try:
                    self.root.after(10, self.actualizar_video)
                except Exception:
                    pass

    def capturar_pagina(self, event=None):
        if self.frame_actual is None:
            # intentar usar último frame disponible en display
            if self.cap is None:
                return
            ret, f = self.cap.read()
            if not ret:
                return
            self.frame_actual = f

        puntos = detectar_libro(self.frame_actual)
        if puntos is None:
            messagebox.showinfo("Aviso", "No se detectó el contorno del libro.")
            return

        imagen_aplanada = aplanar_imagen(self.frame_actual, puntos)
        izq, der = dividir_paginas(imagen_aplanada)

        fecha = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_base = f"{self.var_signatura.get()}_{self.contador:03d}_{fecha}"

        cv2.imwrite(os.path.join(self.output_dir, f"{nombre_base}_L.png"), izq)
        cv2.imwrite(os.path.join(self.output_dir, f"{nombre_base}_R.png"), der)
        self.contador += 1
        messagebox.showinfo("Guardado", f"Páginas guardadas en {self.output_dir}")

    def cerrar(self):
        # Detener el loop y liberar recursos
        self._running = False
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

# -------------------------------
# Inicio del programa
# -------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = BookScannerApp(root)
    root.mainloop()
