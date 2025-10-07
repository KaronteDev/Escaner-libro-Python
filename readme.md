# Escáner de Libros con OpenCV y Tkinter

## Descripción

Aplicación de escritorio en \*\*Python\*\* que utiliza la cámara del ordenador para \*\*escanear libros\*\* o documentos, detectando automáticamente el área del libro, \*\*corrigiendo la perspectiva\*\* y \*\*separando las páginas izquierda y derecha\*\*.

## Características

*   Interfaz gráfica con **Tkinter** 
*   Detección automática del contorno del libro  
*   Corrección de deformación (aplanado de página)  
*   División en dos páginas (izquierda / derecha)  
*   Exportación automática a **PNG numerados**
*   Guardado en carpetas según **signatura** 
*   Captura con tecla **espacio** 
*   Vista previa en tiempo real  
*   Soporte para **metadatos**:  
    *   Título del libro  
    *   Autor  
    *   Signatura (usada como carpeta)  
    *   Archivo / Biblioteca

---

## Requisitos

Python ≥ 3.9    
Librerías necesarias:

Instalación de dependencias

Asegúrate de tener instalados los módulos necesarios:

`pip install opencv-python pillow numpy`

---

## Estructura de salida

```
scans/
 └── SIGNATURA/
      ├── metadata.json
      ├── scan_0001.png
      ├── scan_0002.png
      └── ...
```

El archivo `metadata.json` guarda:

```javascript
{
    "titulo": "Historia de Canarias",
    "autor": "Juan Pérez",
    "archivo": "Archivo Insular de Tenerife",
    "signatura": "AIT-1234" }
```

---

## Ejecuta el programa:

```javascript
python book_scanner.py
```

---

## Generar ejecutable (.exe)

#### 1\. Instala las dependencias

```javascript
pip install -r requirements.txt
```

2\. Instala PyInstaller

Ya está en `requirements.txt`, pero si prefieres hacerlo manualmente:

```javascript
pip install pyinstaller
```

#### 3\. Compila el proyecto

Desde la carpeta donde está `main.py`:

```javascript
pyinstaller --noconfirm --onefile --windowed --name "BookScanner" main.py
```

Esto creará:

```javascript

dist/
 └── BookScanner.exe
build/
BookScanner.spec
```

Puedes mover `BookScanner.exe` a cualquier carpeta, y al ejecutarlo:

*   Se abrirá la **interfaz gráfica**
*   Podrás escanear páginas con **la tecla Espacio**
*   Las imágenes se guardarán en `output/[signatura]/...`

---

## Personalización opcional

Si deseas incluir **icono personalizado** en el ejecutable (por ejemplo `icono.ico`), usa:

```javascript
pyinstaller --noconfirm --onefile --windowed --icon=icono.ico --name "BookScanner" main.py
```

---

## Recomendaciones

*   Usa una **iluminación uniforme** para mejores resultados.
*   Coloca el libro **alineado y centrado** en el campo de visión.
*   Si el detector no encuentra el contorno del libro, ajusta los parámetros del filtro `cv2.Canny` (líneas 29–30 en `main.py`).
*   Puedes escanear indefinidamente: cada vez que pulses **espacio**, se generan dos PNG (izquierda/derecha).
*   Las imágenes se nombran automáticamente con la signatura y número correlativo:

```javascript
SIGNATURA_001_20251007_153000_L.png
SIGNATURA_001_20251007_153000_R.png
```

---

## Cómo usar el `.spec`

1\. Guarda el archivo como `BookScanner.spec` en la **misma carpeta que** `**main.py**`.

2\. Coloca tu **icono** (`icono.ico`) en la misma carpeta (o ajusta la ruta en la línea `icon='icono.ico'`).

3\. Ejecuta PyInstaller con el .spec:

```javascript
pyinstaller BookScanner.spec
```

4\. El ejecutable final estará en:

```javascript
dist/BookScanner/BookScanner.exe
```

5\. Al ejecutarlo:

*   Se abrirá la interfaz GUI.
*   La carpeta `output` se generará automáticamente (según la signatura ingresada).
*   Captura con **espacio**, divide páginas y guarda PNG.