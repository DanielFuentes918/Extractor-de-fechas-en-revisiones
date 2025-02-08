import cv2
import pytesseract
import PyPDF2
import numpy as np
import re
import os
import tkinter as tk
from tkinter import filedialog
from pdf2image import convert_from_path
from PIL import Image

# Configura la ruta de Tesseract y la variable de entorno TESSDATA_PREFIX
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

# Especifica la ruta de Poppler
poppler_path = r"C:\Users\Sistemas2\Downloads\poppler-24.08.0\Library\bin"

# Patrón para fechas en formato dd/mm/yy o dd-mm-yyyy
date_pattern = r'\b(\d{2}[/-]\d{2}[/-]\d{2,4})\b'

def extract_text_from_pdf(pdf_path):
    """
    Extrae texto de un PDF (si es texto seleccionable).
    Retorna un string con el texto extraído.
    """
    text = ""
    with open(pdf_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text

def extract_date(text):
    """Extrae la primera fecha usando expresiones regulares según el patrón date_pattern."""
    matches = re.findall(date_pattern, text)
    return matches[0] if matches else None

def apply_ocr(image, step_name):
    """
    Aplica OCR a la imagen con Tesseract usando una whitelist específica de dígitos y '/' o '-'.
    Guarda la imagen (para depuración), muestra el texto reconocido y busca la fecha.
    Si encuentra la fecha, la devuelve; de lo contrario, retorna None.
    """
    cv2.imwrite(f"processed_{step_name}.png", image)
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="0123456789/-"'
    text = pytesseract.image_to_string(image, config=custom_config).strip()
    
    print(f"\n[OCR en {step_name}]: {text}")
    
    fecha = extract_date(text)
    if fecha:
        print(f"✅ Fecha encontrada en {step_name}: {fecha}")
        return fecha
    return None

def try_contrast_variations(gray):
    """
    Prueba un barrido de valores de contraste (alpha) y brillo (beta).
    Retorna la primera fecha que encuentre, o None si no encuentra nada.
    """
    alphas = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5]
    betas = [-50, -30, 0, 30, 50]
    for alpha in alphas:
        for beta in betas:
            contrasted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
            fecha = apply_ocr(contrasted, f"step_3_contrast_alpha{alpha}_beta{beta}")
            if fecha:
                return fecha
    return None

def try_threshold_variations(gray):
    """
    Prueba diferentes métodos de umbralizado adaptativo (Gaussian) en modo 
    THRESH_BINARY y THRESH_BINARY_INV.
    Retorna la primera fecha que encuentre o None.
    """
    threshold_methods = {
        "binary": cv2.THRESH_BINARY,
        "binary_inv": cv2.THRESH_BINARY_INV
    }
    
    for name, method in threshold_methods.items():
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, method, 11, 2)
        fecha = apply_ocr(thresh, f"step_4_threshold_{name}")
        if fecha:
            return fecha
    return None

def try_noise_reduction_variations(image):
    """
    Prueba diferentes métodos de reducción de ruido:
     - bilateralFilter
     - medianBlur
     - GaussianBlur
    Retorna la primera fecha encontrada, o None si no encuentra nada.
    """
    noise_methods = {
        "bilateral": cv2.bilateralFilter(image, 9, 75, 75),
        "median": cv2.medianBlur(image, 3),
        "gaussian": cv2.GaussianBlur(image, (5, 5), 0)
    }
    
    for name, filtered in noise_methods.items():
        fecha = apply_ocr(filtered, f"step_6_denoised_{name}")
        if fecha:
            return fecha
    return None

# ------------------------------------------------------------------------------
# Función para procesar una imagen que ya tengamos en OpenCV (numpy array)
# ------------------------------------------------------------------------------

def preprocess_and_ocr_from_cv2(image_cv2, debug_prefix="step_pdf"):
    """
    Reutiliza el pipeline de pasos sobre una imagen en formato OpenCV (numpy array).
    Retorna la fecha, o "Fecha no encontrada" si no la detecta.
    """

    if image_cv2 is None:
        print("La imagen de entrada (cv2) es None.")
        return "Fecha no encontrada"

    # 1. Recortar la mitad superior
    h, w, c = image_cv2.shape
    top_half = image_cv2[0:h//2, 0:w]

    # OCR en la imagen original (mitad superior) para ver si encuentra fecha
    fecha = apply_ocr(top_half, f"{debug_prefix}_1_original_top_half")
    if fecha:
        return fecha

    # 2. Convertir a gris
    gray = cv2.cvtColor(top_half, cv2.COLOR_BGR2GRAY)
    fecha = apply_ocr(gray, f"{debug_prefix}_2_grayscale_top_half")
    if fecha:
        return fecha

    # 3. Barrido de contraste
    fecha = try_contrast_variations(gray)
    if fecha:
        return fecha

    # 4. Umbralizado adaptativo
    fecha = try_threshold_variations(gray)
    if fecha:
        return fecha

    # 5. Reducción de ruido
    fecha = try_noise_reduction_variations(gray)
    if fecha:
        return fecha

    # Si nada funcionó, retornamos
    return "Fecha no encontrada"

# ------------------------------------------------------------------------------
# Función para procesar imágenes desde ruta (usa el pipeline anterior)
# ------------------------------------------------------------------------------

def preprocess_and_ocr_from_path(image_path):
    """
    Versión que toma un path de imagen, la carga con cv2 y llama internamente
    a la función preprocess_and_ocr_from_cv2 para hacer el pipeline.
    Retorna la fecha o "Fecha no encontrada".
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"No se pudo leer la imagen: {image_path}")
        return "Fecha no encontrada"
    return preprocess_and_ocr_from_cv2(image, debug_prefix="step_img")

# ------------------------------------------------------------------------------
# Funciones para seleccionar y procesar archivos
# ------------------------------------------------------------------------------

def select_files():
    """Abre un cuadro de diálogo para seleccionar archivos y devuelve sus rutas."""
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Selecciona los archivos PDF o imágenes",
        filetypes=[("Archivos PDF e Imágenes", "*.pdf;*.jpg;*.jpeg;*.png")]
    )
    return file_paths

def process_file(file_path):
    """
    Procesa el archivo dado (PDF o imagen) y extrae la fecha de vencimiento.
    Retorna la fecha como string o "Fecha no encontrada".
    """
    if not file_path:
        print("No se seleccionó ningún archivo.")
        return "Fecha no encontrada"

    # Inicialmente no hemos encontrado fecha
    fecha_vencimiento = "Fecha no encontrada"

    if file_path.lower().endswith(".pdf"):
        # 1) Intentar extraer texto con PyPDF2
        extracted_text = extract_text_from_pdf(file_path)
        fecha_leida = extract_date(extracted_text)
        
        if fecha_leida:
            fecha_vencimiento = fecha_leida
        else:
            # 2) Si no encontramos fecha en el texto, es probable que sea un PDF escaneado.
            #    Convertimos a imágenes y reutilizamos el pipeline de OCR para cada página.
            images = convert_from_path(file_path, dpi=300, poppler_path=poppler_path)
            for i, pil_img in enumerate(images):
                # Convertir de PIL a OpenCV
                open_cv_image = np.array(pil_img)
                # PIL usa RGB, mientras OpenCV trabaja en BGR
                open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

                # Usar el pipeline sobre la imagen en memoria
                fecha_tmp = preprocess_and_ocr_from_cv2(
                    open_cv_image,
                    debug_prefix=f"step_pdf_page_{i}"
                )

                if fecha_tmp != "Fecha no encontrada":
                    fecha_vencimiento = fecha_tmp
                    break
    else:
        # Si el archivo es una imagen, aplicamos el pipeline
        fecha_vencimiento = preprocess_and_ocr_from_path(file_path)
    
    print(f"\n[Resultado para '{file_path}']: {fecha_vencimiento}")
    return fecha_vencimiento

def main():
    # Selección de archivos (PDFs o imágenes)
    file_paths = select_files()
    
    # Lista para guardar los resultados
    results = []
    
    for file_path in file_paths:
        fecha = process_file(file_path)
        # Guardamos en la lista un diccionario con la ruta y la fecha
        results.append({
            "file": file_path,
            "fecha": fecha
        })
    
    # Opcional: al final podrías imprimir todos los resultados juntos
    print("\n--- Resumen de resultados ---")
    for r in results:
        print(f"Archivo: {r['file']} -> Fecha: {r['fecha']}")

if __name__ == "__main__":
    main()
