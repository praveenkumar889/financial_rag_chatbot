import os
import json
import logging
import re
import fitz  # PyMuPDF
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# --------------------------------------------------
# DEFAULT IMAGE INPUT FOLDER
# --------------------------------------------------
DEFAULT_IMAGE_FOLDER = r"C:\financial_agent\images_from_pdf\INFY\Infy_Fy2025"

def run_table_detection(
    image_dir=DEFAULT_IMAGE_FOLDER,
    output_dir=r"C:\financial_agent\outputs\table_detection",
    model_path="yolo26_best.pt",
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

    # --------------------------------------------------
    # CREATE TABLE-ONLY PDF
    # --------------------------------------------------

    table_image_paths = []
    table_page_numbers = set()

    for r in per_image_results:
        if r["tables_detected"] > 0:
            image_name = r["image"]  # page_12.png
            match = re.search(r'page_(\d+)', image_name)
            if match:
                page_num = int(match.group(1))
                table_page_numbers.add(page_num)

                full_img_path = os.path.join(image_dir, image_name)
                table_image_paths.append(full_img_path)

    # Sort images by page number
    table_image_paths = sorted(
        table_image_paths,
        key=lambda x: int(re.search(r'page_(\d+)', x).group(1))
    )

    table_pdf_path = None

    if table_image_paths:

        # 🔥 Extract company name dynamically
        # image_dir = C:\financial_agent\images_from_pdf\INFY\Infy_Fy2025
        # We need to handle potential trailing slashes
        clean_path = os.path.normpath(image_dir)
        parts = clean_path.split(os.sep)

        # Company folder is 2 levels up relative to the image folder name
        # images_from_pdf (previous) -> INFY (interest) -> Infy_Fy2025 (current)
        if len(parts) >= 2:
            company_name = parts[-2]
        else:
            company_name = "Unknown"

        # Ensure input_pdf directory exists
        input_pdf_dir = r"C:\financial_agent\input_pdf"
        os.makedirs(input_pdf_dir, exist_ok=True)

        table_pdf_path = os.path.join(
            input_pdf_dir,
            f"{company_name}_tables.pdf"
        )

        table_doc = fitz.open()

        for img_path in table_image_paths:
            img = fitz.open(img_path)
            pdf_bytes = img.convert_to_pdf()
            img_pdf = fitz.open("pdf", pdf_bytes)
            table_doc.insert_pdf(img_pdf)
            img.close()

        # Save with garbage collection/deflate to keep it clean
        table_doc.save(table_pdf_path, garbage=4, deflate=True)
        table_doc.close()

        print(f"📄 Table PDF created at: {table_pdf_path}")
        print(f"📌 Table Pages Detected: {sorted(table_page_numbers)}")

    else:
        print("⚠️ No tables detected.")

    return per_image_results, summary_metrics, table_page_numbers, table_pdf_path

if __name__ == "__main__":
    print("🚀 Running Table Detection...")

    # Ensure model path is correct based on where script is run
    # If script run from c:\financial_agent\imagestopdf_tabledetection, it's just filename
    # If run from root, might need full path.  Using default argument "yolo26_best.pt" assumes CWD.
    
    try:
        results, metrics, pages, pdf_path = run_table_detection()
        print("✅ Done.")
    except Exception as e:
        print(f"❌ Error: {e}")
