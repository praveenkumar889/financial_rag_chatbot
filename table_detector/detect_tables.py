import camelot
import sys
import os
import json
from pathlib import Path
import fitz  # pymupdf

# --- CONFIGURATION: SET YOUR INPUT FILE HERE ---
# You can paste your PDF path below to avoid typing it in the terminal every time.
DEFAULT_PDF_PATH = r"C:\financial_agent\table_pdf\INFY_FY2025.pdf"
# -----------------------------------------------

def detect_tables(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return

    print(f"Scanning {pdf_path} for tables...") 
    print("Using Camelot 'lattice' mode (best for solid grid tables)...")
    print("This allows us to identify pages that should be skipped/handled by Textract.")

    try:
        # Get total page count first using PyMuPDF (fast)
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        print(f"Total Pages to Scan: {total_pages}")

        # 1. SCAN FOR TABLES
        print("Scanning for tables (Lattice Mode)...")
        # flavor='lattice' detects grid-like tables. 
        # flavor='stream' detects whitespace tables (less reliable for mixed content).
        try:
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
        except Exception as e:
            print(f"Camelot scan failed: {e}")
            return

        # Map page -> list of table bboxes (camelot coordinates)
        # Camelot bbox: (x0, y0, x1, y1) where y is from Bottom-Left
        page_tables = {}
        for table in tables:
            p = int(table.page)
            if p not in page_tables:
                page_tables[p] = []
            page_tables[p].append(table._bbox)

        # 2. ANALYZE PAGE CONTENT (Mix Detection)
        print("Analyzing page content for mixed narrative...")
        
        narrative_pages = []
        mixed_pages = []      # Table + Narrative
        table_only_pages = [] # Mostly Table
        
        doc = fitz.open(pdf_path)
        
        for page_num in range(1, total_pages + 1):
            if page_num not in page_tables:
                narrative_pages.append(page_num)
                continue
                
            # Page has tables, determine if it's Mixed or Table-Only
            page = doc[page_num - 1]
            page_height = page.rect.height
            
            # Get all text blocks
            text_blocks = page.get_text("blocks") # (x0, y0, x1, y1, text, block_no, block_type)
            
            non_table_text_len = 0
            
            # Convert Camelot bboxes to Fitz bboxes
            # Camelot: (x0, y0, x1, y1) -> y from bottom
            # Fitz: (x0, y0, x1, y1) -> y from top
            
            fitz_table_rects = []
            for c_bbox in page_tables[page_num]:
                c_x0, c_y0, c_x1, c_y1 = c_bbox
                # Flip Y axis
                f_y0 = page_height - c_y1
                f_y1 = page_height - c_y0
                fitz_table_rects.append(fitz.Rect(c_x0, f_y0, c_x1, f_y1))
            
            # Check overlapping text
            for b in text_blocks:
                b_rect = fitz.Rect(b[:4])
                text = b[4].strip()
                if not text: continue
                
                # Check if this block is largely inside any table
                is_inside_table = False
                for t_rect in fitz_table_rects:
                    # If intersection area is significant, treat as table text
                    # Simple check: if center of text block is in table rect
                    center = b_rect.tl + (b_rect.br - b_rect.tl) * 0.5
                    if t_rect.contains(center):
                        is_inside_table = True
                        break
                
                if not is_inside_table:
                    non_table_text_len += len(text)
            
            # Threshold: If > 200 characters of non-table text (~30-40 words), it's Mixed
            # This accounts for headers/footers which might add ~50-100 chars.
            # Let's set a safe threshold for "meaningful narrative"
            if non_table_text_len > 150:
                mixed_pages.append(page_num)
            else:
                table_only_pages.append(page_num)

        doc.close()
        
        narrative_pages.sort()
        mixed_pages.sort()
        table_only_pages.sort()
        
        # print outputs to console
        print(f"\n--- SCAN COMPLETE ---")
        print(f"Narrative-only pages ({len(narrative_pages)}): {narrative_pages}")
        print(f"Table-Plain Data (Mixed) ({len(mixed_pages)}): {mixed_pages}")
        print(f"Pages with Tables (Mostly Table) ({len(table_only_pages)}): {table_only_pages}")

        # --- SAVE OUTPUT ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 1. Text Report
        txt_output = os.path.join(script_dir, "scan_results.txt")
        with open(txt_output, "w", encoding="utf-8") as f:
            f.write(f"PDF Table Scan Report\n")
            f.write(f"File: {pdf_path}\n")
            f.write(f"Total Pages: {total_pages}\n")
            f.write(f"--------------------------------------------------\n")
            f.write(f"Narrative-only pages: {narrative_pages}\n\n")
            f.write(f"Table-Plain Data (Mixed): {mixed_pages}\n\n")
            f.write(f"Pages with Tables (Only): {table_only_pages}\n")
            
        # 2. JSON Report
        json_output = os.path.join(script_dir, "scan_results.json")
        data = {
            "source_file": pdf_path,
            "total_pages": total_pages,
            "narrative_pages": narrative_pages,
            "mixed_pages": mixed_pages,
            "table_only_pages": table_only_pages
        }
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        print(f"\nResults saved successfully to:")
        print(f"1. {txt_output}")
        print(f"2. {json_output}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        if "Ghostscript" in str(e):
            print("\nCRITICAL DEPENDENCY ERROR: Ghostscript is missing.")
            print("Camelot requires Ghostscript to read PDFs.")
            print("Please install it from: https://www.ghostscript.com/download/gsdnld.html")
            print("Restart your terminal after installation.")

if __name__ == "__main__":
    # Priority: 1. Command Line Argument, 2. Default Path
    if len(sys.argv) > 1:
        target_pdf = sys.argv[1]
    else:
        target_pdf = DEFAULT_PDF_PATH
    
    # Strip potential quotes from drag-and-drop
    target_pdf = target_pdf.strip('"').strip("'")
    
    detect_tables(target_pdf)
