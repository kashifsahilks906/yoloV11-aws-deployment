import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np

model_loaded = YOLO('best.pt')

def predict(image):

    image_bytes = np.frombuffer(image.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    results = model_loaded(image, conf=0.50)[0]

    detections = sv.Detections(
        xyxy=results.boxes.xyxy.cpu().numpy(),
        confidence=results.boxes.conf.cpu().numpy(),
        class_id=results.boxes.cls.cpu().numpy().astype(int)
    )

    class_names = model_loaded.names

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=1
    )

    annotated_image = box_annotator.annotate(scene=image, detections=detections)

    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=detections,
        labels=[f"{class_names[class_id]} {confidence:.2f}" 
                for class_id, confidence in zip(detections.class_id, detections.confidence)]
    )

    _, buffer = cv2.imencode('.jpg', annotated_image)

    results_list = []
    for box, conf, cls in zip(detections.xyxy, detections.confidence, detections.class_id):
        results_list.append(
            f"Detection: {class_names[cls]}, Confidence: {conf:.2f}, Coordinates: {box}"
        )

    return buffer, results_list