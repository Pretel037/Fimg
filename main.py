import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO

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

# Interfaz de usuario en Streamlit
st.title('Generador de Imágenes con Stable Diffusion')

prompt = st.text_input("Introduce el prompt para generar la imagen:")

if st.button('Generar Imagenes'):
    if prompt:
        imagenes_generadas = generar_imagenes_con_stable_diffusion(prompt, num_imagenes=3)
        
        # Exponer las imágenes generadas para que otros servicios puedan acceder a ellas
        imagenes_urls = []
        for idx, img_byte_arr in enumerate(imagenes_generadas):
            imagenes_urls.append(img_byte_arr)
            st.image(img_byte_arr, caption=f"Imagen {idx + 1}")
        
        st.success("Las imágenes han sido generadas.")
