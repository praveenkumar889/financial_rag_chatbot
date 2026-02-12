from imagestopdf_tabledetection.table_detection import run_table_detection
from data_extraction.financial_extraction import run_extraction
from imagestopdf_tabledetection.pdf_to_images import convert_pdf_to_images
import os
import shutil

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
SELECTED_PDF = r"C:\financial_agent\input_pdf\INFY_Fy2025.pdf"
IMAGES_ROOT = r"C:\financial_agent\images_from_pdf"
MODEL_PATH = r"C:\financial_agent\imagestopdf_tabledetection\yolo26_best.pt"
OUTPUT_NARRATIVE = r"C:\financial_agent\outputs\financial_extraction"

def main():

    # --------------------------------------------------
    # Step 0: PDF to Images
    # --------------------------------------------------
    print(f"\n🚀 Step 0: Converting PDF to Images: {os.path.basename(SELECTED_PDF)}")
    try:
        # This returns the folder where images were saved (e.g. .../INFY/Infy_Fy2025)
        image_folder = convert_pdf_to_images(
            pdf_path=SELECTED_PDF, 
            output_root=IMAGES_ROOT
        )
        print(f"✅ Images saved at: {image_folder}")
    except Exception as e:
        print(f"❌ Image conversion failed: {e}")
        return

    # --------------------------------------------------
    # Step 1: Table Detection
    # --------------------------------------------------
    print(f"\n🚀 Step 1: Running Table Detection on {os.path.basename(image_folder)}")
    try:
        results, metrics, detected_table_pages, table_pdf_path = run_table_detection(
            image_dir=image_folder,
            model_path=MODEL_PATH
        )
    except Exception as e:
        print(f"❌ Table detection failed: {e}")
        return

    print("📌 Detected Table Pages:", sorted(detected_table_pages))

    # --------------------------------------------------
    # Step 2: Narrative Extraction
    # --------------------------------------------------
    print(f"\n🚀 Step 2: Running Narrative Extraction for {os.path.basename(SELECTED_PDF)}")
    try:
        run_extraction(
            pdf_path=SELECTED_PDF,
            skip_pages=detected_table_pages,
            output_folder=OUTPUT_NARRATIVE
        )
    except Exception as e:
         print(f"❌ Narrative extraction failed: {e}")
         return

    # --------------------------------------------------
    # Step 3: Cleanup
    # --------------------------------------------------
    print(f"\n🚀 Step 3: Cleaning up temporary images...")
    try:
        if os.path.exists(image_folder) and IMAGES_ROOT in os.path.abspath(image_folder):
            shutil.rmtree(image_folder)
            print(f"✅ Deleted temporary image folder: {image_folder}")
        else:
            print(f"⚠️ Skipped deletion (safety check): {image_folder}")
    except Exception as e:
        print(f"⚠️ Cleanup failed: {e}")

    print("\n✅ Full Pipeline Completed Successfully")


if __name__ == "__main__":
    main()
