import os
import json
import logging
from ultralytics import YOLO
from ultralytics.utils import LOGGER

def run_table_detection(
    image_dir,
    output_dir,
    model_path,
    conf=0.4
):
    """
    Runs YOLO table detection on images in image_dir.
    Saves marked images in output_dir/marked_images and a JSON log in output_dir.
    """
    # --------------------------------------------------
    # Paths (Organize by PDF name to avoid collisions)
    # --------------------------------------------------
    pdf_name = os.path.basename(image_dir)
    pdf_output_dir = os.path.join(output_dir, pdf_name)
    marked_images_dir = os.path.join(pdf_output_dir, "marked_images")
    
    os.makedirs(marked_images_dir, exist_ok=True)
    
    log_path = os.path.join(pdf_output_dir, "log.json")
    txt_log_path = os.path.join(pdf_output_dir, "predict.log")

    # ----------------------------
    # Attach YOLO logger to a file
    # ----------------------------
    file_handler = logging.FileHandler(txt_log_path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(file_handler)

    # ----------------------------
    # Load model once
    # ----------------------------
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    model = YOLO(model_path)

    # ----------------------------
    # Run prediction
    # ----------------------------
    # We set save=False and manually save only if boxes are detected
    results = model.predict(
        source=image_dir,
        conf=conf,
        save=False,
        show_labels=True,
        show_conf=True,
        verbose=True
    )

    LOGGER.removeHandler(file_handler)

    # --------------------------------------------------
    # Structured outputs
    # --------------------------------------------------
    per_image_results = []
    detections_per_class = {}
    all_confidences = []

    for r in results:
        image_name = os.path.basename(r.path)
        h, w = r.orig_shape
        resolution = f"{w}x{h}"

        boxes = []
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf_score = float(box.conf[0])

            boxes.append({
                "class": cls_name,
                "confidence": conf_score,
                "bbox": box.xyxy[0].tolist() # [xmin, ymin, xmax, ymax]
            })

            detections_per_class[cls_name] = detections_per_class.get(cls_name, 0) + 1
            all_confidences.append(conf_score)

        num_tables = len(boxes)

        # Manually save the marked image ONLY if at least one table is detected
        if num_tables > 0:
            marked_image_path = os.path.join(marked_images_dir, image_name)
            r.save(filename=marked_image_path)

        speed = r.speed

        per_image_results.append({
            "image": image_name,
            "resolution": resolution,
            "tables_detected": num_tables,
            "speed": {k: round(v, 2) for k, v in speed.items()},
            "detections": boxes
        })

    # --------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------
    summary_metrics = {
        "total_images": len(per_image_results),
        "images_with_tables": sum(1 for r in per_image_results if r["tables_detected"] > 0),
        "total_tables": sum(r["tables_detected"] for r in per_image_results),
        "detections_per_class": detections_per_class,
        "average_confidence": (sum(all_confidences) / len(all_confidences) if all_confidences else 0.0)
    }

    final_output = {
        "summary": summary_metrics,
        "results": per_image_results
    }

    # --------------------------------------------------
    # Save JSON Log
    # --------------------------------------------------
    with open(log_path, "w") as f:
        json.dump(final_output, f, indent=4)

    print(f"✅ Table detection complete for {pdf_name}")
    print(f"📂 Only images with tables were saved to: {marked_images_dir}")
    print(f"📊 JSON log saved to: {log_path}")

    return per_image_results, summary_metrics
