import os
import cv2
import json
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pycocotools.coco import COCO

# Configuración de rutas
ANNOTATIONS_PATH = "profundo/dataset/annotations/instances_val2017.json"
IMAGES_DIR = "profundo/dataset/val2017/"
IMAGE_ID = 139
IMAGE_FILENAME = f"{IMAGE_ID:012d}.jpg"  # COCO usa 12 ceros de padding
IMAGE_PATH = os.path.join(IMAGES_DIR, IMAGE_FILENAME)
OUTPUT_DIR = "profundo/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_iou(box1, box2):
    """Calcula Intersection over Union entre dos boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def get_ground_truth(coco, img_id):
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    gt_boxes = []
    gt_classes = []
    for ann in anns:
        # COCO format: [x, y, width, height] -> Convertir a [x1, y1, x2, y2]
        x, y, w, h = ann['bbox']
        gt_boxes.append([x, y, x + w, y + h])
        gt_classes.append(coco.loadCats(ann['category_id'])[0]['name'])
    return gt_boxes, gt_classes

def draw_boxes(image, boxes, labels, color, thickness=2):
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(image, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    return image

def main():
    # 1. Cargar COCO
    coco = COCO(ANNOTATIONS_PATH)
    gt_boxes, gt_classes = get_ground_truth(coco, IMAGE_ID)
    
    # Visualizar Ground Truth inicial
    img_gt = cv2.imread(IMAGE_PATH)
    img_gt_viz = draw_boxes(img_gt.copy(), gt_boxes, gt_classes, (0, 255, 0)) # Verde para GT
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ground_truth_only.jpg"), img_gt_viz)

    # 2. Benchmark de modelos YOLO
    model_variants = ['n', 's', 'm', 'l', 'x']
    results_data = []
    all_predictions = {}

    for var in model_variants:
        model_name = f"yolov8{var}.pt"
        model = YOLO(model_name)
        results = model(IMAGE_PATH, verbose=False)[0]
        
        pred_boxes = results.boxes.xyxy.cpu().numpy()
        pred_conf = results.boxes.conf.cpu().numpy()
        pred_cls = [results.names[int(c)] for c in results.boxes.cls.cpu().numpy()]
        
        all_predictions[var] = {
            'boxes': pred_boxes,
            'classes': pred_cls,
            'conf': pred_conf
        }

        # Calcular métricas simples para esta imagen (IoU > 0.5)
        tp = 0
        matched_gt = set()
        for p_box in pred_boxes:
            for i, g_box in enumerate(gt_boxes):
                if i not in matched_gt and calculate_iou(p_box, g_box) > 0.5:
                    tp += 1
                    matched_gt.add(i)
                    break
        
        precision = tp / len(pred_boxes) if len(pred_boxes) > 0 else 0
        recall = tp / len(gt_boxes) if len(gt_boxes) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results_data.append({
            "Modelo": model_name,
            "Detecciones": len(pred_boxes),
            "Conf_Media": np.mean(pred_conf) if len(pred_conf) > 0 else 0,
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1-Score": round(f1, 3)
        })

    # 3. Mostrar Tabla Comparativa
    df = pd.DataFrame(results_data)
    print("\n--- Comparativa de Modelos YOLO (Imagen 139) ---")
    print(df.to_string(index=False))

    # 4. Seleccionar el modelo con más detecciones
    best_var = df.loc[df['Detecciones'].idxmax()]['Modelo'].split('v8')[1].split('.')[0]
    best_preds = all_predictions[best_var]
    
    print(f"\nEl modelo con más detecciones es: yolov8{best_var}")
    
    # 5. Visualización Final (GT vs Preds)
    final_viz = cv2.imread(IMAGE_PATH)
    # Dibujar Ground Truth en Verde
    final_viz = draw_boxes(final_viz, gt_boxes, gt_classes, (0, 255, 0), 2)
    # Dibujar Predicciones en Rojo
    pred_labels = [f"{c} {conf:.2f}" for c, conf in zip(best_preds['classes'], best_preds['conf'])]
    final_viz = draw_boxes(final_viz, best_preds['boxes'], pred_labels, (0, 0, 255), 1)
    
    # Guardar resultado
    output_path = os.path.join(OUTPUT_DIR, f"final_comparison_yolov8{best_var}.jpg")
    cv2.imwrite(output_path, final_viz)
    print(f"Resultado visual guardado en: {output_path}")

if __name__ == "__main__":
    main()