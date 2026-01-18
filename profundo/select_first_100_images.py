# This file is used to select the first 100 
# images from the full dataset.

import json
import os
import shutil

# --- CONFIGURACIÓN DE RUTAS ---
ruta_json_original = 'full_dataset/annotations/instances_val2017.json'
ruta_imagenes_original = 'full_dataset/val2017/'

ruta_json_destino = 'dataset/annotations/'
ruta_imagenes_destino = 'dataset/val2017/'

# Crear carpeta de destino si no existe
os.makedirs(ruta_imagenes_destino, exist_ok=True)

# 1. Cargar el archivo JSON original
print("Cargando anotaciones originales...")
with open(ruta_json_original, 'r') as f:
    data = json.load(f)

# 2. Seleccionar las primeras 100 imágenes
subset_images = data['images'][:100]
subset_img_ids = set(img['id'] for img in subset_images)

# 3. Filtrar las anotaciones que pertenecen a esas 100 imágenes
subset_annotations = [ann for ann in data['annotations'] if ann['image_id'] in subset_img_ids]

# 4. Crear el nuevo diccionario con la misma estructura de COCO
subset_data = {
    "info": data.get("info", {}),
    "licenses": data.get("licenses", []),
    "images": subset_images,
    "annotations": subset_annotations,
    "categories": data.get("categories", [])
}

# 5. Guardar el nuevo archivo JSON
with open(ruta_json_destino, 'w') as f:
    json.dump(subset_data, f)
print(f"Nuevo JSON guardado en: {ruta_json_destino}")

# 6. Copiar los archivos de imagen físicamente
print("Copiando imágenes...")
count = 0
for img in subset_images:
    nombre_archivo = img['file_name']
    path_origen = os.path.join(ruta_imagenes_original, nombre_archivo)
    path_destino = os.path.join(ruta_imagenes_destino, nombre_archivo)
    
    if os.path.exists(path_origen):
        shutil.copy(path_origen, path_destino)
        count += 1
    else:
        print(f"Advertencia: No se encontró la imagen {nombre_archivo}")

print(f"Proceso finalizado. Se han copiado {count} imágenes a {ruta_imagenes_destino}")

