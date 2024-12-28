from flask import Flask, request, send_file, jsonify
from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO

app = Flask(__name__)

# Función para generar imágenes con Stable Diffusion
def generar_imagenes_con_stable_diffusion(prompt, num_imagenes=1):
    """
    Genera imágenes utilizando Stable Diffusion a partir de un prompt.
    
    Parámetros:
    - prompt (str): Descripción de la imagen a generar.
    - num_imagenes (int): Número de imágenes a generar.
    """
    # Cargar el modelo preentrenado de Stable Diffusion
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    
    imagenes_generadas = []
    
    for i in range(num_imagenes):
        # Generar la imagen
        image = pipe(prompt).images[0]
        
        # Guardar la imagen en un objeto BytesIO
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        imagenes_generadas.append(img_byte_arr)
    
    return imagenes_generadas

@app.route('/generar_imagen', methods=['POST'])
def generar_imagen():
    prompt = request.json.get('prompt')
    
    if not prompt:
        return jsonify({"error": "El prompt es obligatorio"}), 400

    # Generar las imágenes
    imagenes_generadas = generar_imagenes_con_stable_diffusion(prompt, num_imagenes=1)

    # Se asume que solo se genera una imagen en este caso
    img_byte_arr = imagenes_generadas[0]

    # Devolver la imagen generada como un archivo descargable
    return send_file(
        img_byte_arr,
        mimetype='image/png',
        as_attachment=True,
        download_name="imagen_generada.png"
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
