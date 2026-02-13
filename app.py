import os
import shutil
import traceback
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from rag_core.engine import answer_question

# Imports for Orchestrator
from imagestopdf_tabledetection.pdf_to_images import convert_pdf_to_images
from imagestopdf_tabledetection.table_detection import run_table_detection
from data_extraction.financial_extraction import run_extraction as run_financial_extraction
from data_extraction.transcript_extraction import run_extraction as run_transcript_extraction
from data_extraction.ppt_extraction import run_ppt_extraction
from data_extraction.table_extraction import run_table_extraction
from rag_ingestion.ingest import run_ingestion

# Create FastAPI app
app = FastAPI(title="Financial RAG API")

# Configuration
IMAGES_ROOT = "images_from_pdf"
OUTPUT_NARRATIVE = "outputs"
YOLO_MODEL_PATH = os.path.join("imagestopdf_tabledetection", "yolo26_best.pt")
INPUT_PDF_DIR = "input_pdf"

os.makedirs(INPUT_PDF_DIR, exist_ok=True)
os.makedirs(OUTPUT_NARRATIVE, exist_ok=True)

# Request schema
class QueryRequest(BaseModel):
    question: str

# Health endpoint
@app.get("/")
def health_check():
    return {"status": "RAG API is running"}

# Main RAG endpoint
@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        response = answer_question(request.question)
        # Return answer AND structured sources
        return {
            "answer": response.get("answer", "No answer found."),
            "sources": response.get("sources", [])
        }
    except Exception as e:
        return {"error": str(e)}

# -----------------------------------------------------------------------------
# 🧠 INTELLIGENT DOCUMENT ORCHESTRATOR
# -----------------------------------------------------------------------------

def detect_document_type(filename: str):
    name = filename.lower()
    if "transcript" in name:
        return "TRANSCRIPT"
    elif "ppt" in name or "presentation" in name:
        return "PRESENTATION"
    elif "table" in name:
        return "TABLE"
    else:
        return "FINANCIAL_REPORT"

def extract_metadata(filename: str):
    name = filename.replace(".pdf", "")
    clean_name = name.replace("-", "_")
    parts = clean_name.split("_")

    company = parts[0]
    year = "Unknown"
    quarter = "Unknown"

    for p in parts:
        if p.lower().startswith("fy") or (p.isdigit() and len(p) == 4):
            year = p
        if p.lower().startswith("q") and len(p) == 2:
            quarter = p

    return company, year, quarter

def process_document(pdf_path: str):
    filename = os.path.basename(pdf_path)
    doc_type = detect_document_type(filename)
    company, year, quarter = extract_metadata(filename)
    
    # Specific subfolder for clean outputs
    output_subfolder = os.path.join(OUTPUT_NARRATIVE, doc_type.lower() + "_extraction")
    os.makedirs(output_subfolder, exist_ok=True)

    output_txt = None

    try:
        if doc_type == "FINANCIAL_REPORT":
            # 1. Convert to Images
            image_folder = convert_pdf_to_images(pdf_path, IMAGES_ROOT)
            if not image_folder:
                 raise Exception("Image conversion failed")

            # 2. Table Detection
            _, _, table_pages, _ = run_table_detection(
                image_dir=image_folder,
                model_path=YOLO_MODEL_PATH
            )
            
            # 3. Narrative Extraction (Skip Tables)
            output_txt = run_financial_extraction(
                pdf_path=pdf_path,
                skip_pages=table_pages,
                output_folder=output_subfolder
            )

        elif doc_type == "TRANSCRIPT":
             output_txt = run_transcript_extraction(
                pdf_path=pdf_path,
                output_folder=output_subfolder
            )

        elif doc_type == "PRESENTATION":
             # PPT Extractor saves to its own logic, we pass the folder
             output_txt = run_ppt_extraction(
                pdf_path=pdf_path,
                output_dir=output_subfolder
            )
            
        elif doc_type == "TABLE":
             output_txt = run_table_extraction(
                pdf_path=pdf_path,
                output_folder=output_subfolder
            )

        if not output_txt:
            raise Exception("Extraction failed to produce an output file.")

        # 4. Ingest to MongoDB
        run_ingestion(
            file_path=output_txt,
            company=company,
            year=year,
            quarter=quarter,
            doc_type=doc_type
        )

        return {
            "status": "success",
            "filename": filename,
            "company": company,
            "year": year,
            "quarter": quarter,
            "doc_type": doc_type,
            "extraction_path": output_txt
        }
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/upload-and-process")
async def upload_and_process(file: UploadFile = File(...)):
    # Save Uploaded File
    save_path = os.path.join(INPUT_PDF_DIR, file.filename)
    
    try:
        with open(save_path, "wb") as f:
            f.write(await file.read())
            
        print(f"File saved to {save_path}")
        
        # Trigger Processing
        result = process_document(save_path)
        
        return result
        
    except Exception as e:
         return {"error": str(e)}
