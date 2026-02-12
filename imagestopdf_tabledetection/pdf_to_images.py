import fitz  # PyMuPDF
import os
import re

def convert_pdf_to_images(pdf_path, output_root, page_numbers=None):
    """
    Converts PDF pages into images and prints conversion progress.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    # Auto-generate subfolder from PDF name
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Simple heuristic: Company name is the first part of the filename (e.g., INFY_FY2025 -> INFY)
    # Adjust this logic if your naming convention differs
    company_name = pdf_name.split('_')[0].upper()
    
    folder_name = re.sub(r'[^a-zA-Z0-9]', ' ', pdf_name).title().replace(' ', '_')
    
    # Structure: output_root / Company / DocumentName
    output_folder = os.path.join(output_root, company_name, folder_name)

    os.makedirs(output_folder, exist_ok=True)
    print(f"--- Converting PDF to Images: {os.path.basename(pdf_path)} ---")
    print(f"Target folder: {output_folder}")

    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        if page_numbers is None:
            pages_to_convert = list(range(total_pages))
            print(f"🔄 Processing all {total_pages} pages...")
        else:
            pages_to_convert = [p for p in page_numbers if 0 <= p < total_pages]
            if not pages_to_convert:
                print("❌ No valid pages found for conversion.")
                return None
            print(f"🔄 Processing specific pages: {pages_to_convert}")

        zoom = 300 / 72  # 300 DPI
        matrix = fitz.Matrix(zoom, zoom)

        for i, page_index in enumerate(pages_to_convert, start=1):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image_name = f"page_{page_index + 1}.png"
            image_path = os.path.join(output_folder, image_name)
            pix.save(image_path)
            print(f"  [{i}/{len(pages_to_convert)}] Saved {image_name}")

        doc.close()
        print(f"✅ Conversion complete. Images saved in: {folder_name}")
        return output_folder

    except Exception as e:
        print(f"Error during image conversion: {e}")
        raise RuntimeError(f"Conversion failed: {str(e)}")

if __name__ == "__main__":
    # ---------------------------------------------------------
    # ⚙️ CONFIGURATION: SET YOUR PDF PATH HERE
    # ---------------------------------------------------------
    pdf_path = r"C:\financial_agent\input_pdf\INFY_FY2025.pdf"
    output_root = r"C:\financial_agent\images_from_pdf"
    
    # Optional: extract specific pages (e.g. [0, 1, 5]) or None for all
    pages_to_extract = None 
    
    convert_pdf_to_images(pdf_path, output_root, page_numbers=pages_to_extract)