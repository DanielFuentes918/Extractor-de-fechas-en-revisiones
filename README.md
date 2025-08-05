# Script de Detección Automática de Fechas en PDFs e Imágenes

Este script automatiza la detección de fechas en documentos PDF (ya sean digitales o escaneados) e imágenes. Utiliza procesamiento de imágenes y OCR mediante Tesseract para encontrar fechas escritas en diversos formatos. Está diseñado para facilitar tareas de lectura automatizada de vencimientos u otras fechas relevantes en documentos.

## 🚀 Características principales

- Soporte para PDFs con texto seleccionable y PDFs escaneados.
- Detección automática de fechas en formato `dd/mm/yyyy`, `dd-mm-yy`, etc.
- Pipeline de procesamiento robusto con múltiples pasos:
  - Conversión a escala de grises
  - Variaciones de contraste y brillo
  - Umbralizado adaptativo
  - Reducción de ruido
  - OCR con Tesseract configurado para priorizar fechas.
- Interfaz gráfica para seleccionar archivos desde el explorador (Tkinter).
- Procesamiento en lote con impresión de resumen final.

## ⚙️ Requisitos

- Python 3.8 o superior
- Tesseract OCR instalado
- Poppler (en Windows) para convertir PDFs escaneados en imágenes

### Librerías de Python necesarias

- opencv-python
- pytesseract
- PyPDF2
- numpy
- pdf2image
- pillow
- tkinter (incluido en la mayoría de instalaciones de Python en Windows)
