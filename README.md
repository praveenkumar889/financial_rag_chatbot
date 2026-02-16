# AI Financial Analysis Agent - README

## 🚀 Overview
The **AI Financial Analysis Agent** is an advanced Retrieval-Augmented Generation (RAG) system designed to process, analyze, and query complex financial documents. It supports multi-modal data ingestion (PDFs, PPTs, Tables, Transcripts), strict multi-company isolation, and intelligent query handling with source traceability.

## 🏗️ Architecture
The system follows a modular pipeline:
1.  **Ingestion & Extraction**:
    *   **Financial Reports**: Converts PDFs to images, detects tables using YOLO, extracts narrative text while preserving layout.
    *   **Transcripts**: Speaker-aware extraction with role detection (Analyst vs. Management).
    *   **PPTs**: Visual layout analysis using Azure OpenAI GPT-4 Vision.
    *   **Tables**: High-precision extraction using AWS Textract with semantic narrative generation.
2.  **Vector Database**: MongoDB Atlas with vector search indices for retrieved chunks.
3.  **RAG Engine (`rag_core/engine.py`)**:
    *   **Dynamic Isolation**: Auto-detects companies and enforces strict data separation.
    *   **Hybrid Search**: Combines vector similarity with metadata filtering (Company, Year, Doc Type).
    *   **Self-Correction**: LLM-based critique loop to ensure factual accuracy (prevents hallucinations).
4.  **API Layer**: FastAPI backend with `uvicorn` server.

---

## ⚡ Quick Start (Copy & Paste)
Use these commands to quickly set up and run the application on **Windows**.

### 1. First Time Setup (Automated)
Run the automated setup script to install Python 3.11 environment and dependencies:
```powershell
.\setup_project.ps1
```

### 1b. Manual Setup (Alternative)
If you prefer manual setup:
```powershell
# 1. Clone the repository
git clone <your-repo-url>
cd financial_agent

# 2. Check Python Version (Must be 3.11.x)
python --version
# If you have the py launcher:
# py -3.11 --version

# 3. Create Virtual Environment with Python 3.11
# Windows Launcher (recommended)
py -3.11 -m venv venv

# 4. Activate Virtual Environment
.\venv\Scripts\activate

# 5. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root `financial_agent` folder:
```ini
# Azure OpenAI
AZURE_AI_ENDPOINT="https://your-resource.openai.azure.com/"
AZURE_AI_API_KEY="your-api-key"
AZURE_EMBEDDING_DEPLOYMENT="text-embedding-ada-002"
GPT_4_1_MINI_DEPLOYMENT="gpt-4"

# MongoDB
MONGO_URI="mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true&w=majority"
MONGO_DB_NAME="financial_rag"
MONGO_COLLECTION_NAME="rag_chunks"

# AWS (For Textract)
AWS_ACCESS_KEY_ID="your-aws-key"
AWS_SECRET_ACCESS_KEY="your-aws-secret"
S3_BUCKET_NAME="your-s3-bucket"
AWS_REGION="ap-south-1"
```

### 3. Run the Application
```powershell
# 1. Activate Environment (if not already active)
.\venv\Scripts\activate

# 2. Start the API Server
uvicorn app:app --reload
```
*The API will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)*

---

## 🔄 Full Data Pipeline Workflow

If you want to process new documents (PDFs) from scratch, follow this pipeline.

### Step 1: Run Extractors (Data Extraction)

**A. Financial Reports (Annual Reports)**
Extracts narrative text and detects tables to avoid contamination.
```powershell
.\venv\Scripts\activate
python data_extraction/financial_extraction.py
```
*Input*: `input_pdf/INFY_FY2025.pdf`
*Output*: `outputs/financial_extraction/INFY_FY2025.txt`

**B. Earnings Call Transcripts**
Extracts speaker-diarized text.
```powershell
.\venv\Scripts\activate
python data_extraction/transcript_extraction.py
```
*Input*: `input_pdf/INFY_2025_transcript.pdf`
*Output*: `outputs/transcript_extraction/INFY_2025_transcript.txt`

**C. Presentation Decks (PPTs)**
Uses GPT-4 Vision to analyze slides.
```powershell
.\venv\Scripts\activate
python data_extraction/ppt_extraction.py "input_pdf/infy-ppt_Q3_FY26.pdf"
```
*Output*: `outputs/ppt_extraction/infy-ppt_Q3_FY26_narrative.txt`

**D. Table Extraction (AWS Textract)**
Extracts complex tables and converts them to semantic text.
```powershell
.\venv\Scripts\activate
python data_extraction/table_extraction.py "input_pdf/INFY_Tables.pdf"
```

### Step 2: Ingest to MongoDB (Vector Store)
Once extraction is complete, ingest the processed text files into MongoDB.
```powershell
.\venv\Scripts\activate
python rag_ingestion/ingest.py
```
*This script scans all `outputs/` folders and uploads chunks + embeddings to MongoDB.*

---

## 🛠️ Testing the API

Once the server is running (`uvicorn app:app --reload`), you can test it:

**Option 1: Swagger UI**
1.  Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
2.  Click **POST /ask** -> **Try it out**
3.  Enter JSON:
    ```json
    {
      "question": "What is the operating margin for Infosys in FY24?"
    }
    ```
4.  Click **Execute**.

**Option 2: PowerShell / Curl**
```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/ask" -ContentType "application/json" -Body '{"question": "What is the operating margin?"}'
```

---

## 📂 Project Structure

```
c:/financial_agent/
├── app.py                      # FastAPI Backend Entry Point
├── rag_core/
│   ├── engine.py               # RAG Logic (Retrieval, Filtering, LLM Generation)
├── rag_ingestion/
│   ├── ingest.py               # MongoDB Ingestion Script
├── data_extraction/
│   ├── financial_extraction.py # PDF Annual Report Extractor
│   ├── transcript_extraction.py# Speaker Verification & Transcript Extractor
│   ├── ppt_extraction.py       # GPT-4 Vision PPT Analyzer
│   ├── table_extraction.py     # AWS Textract Table Processor
├── imagestopdf_tabledetection/
│   ├── pdf_to_images.py        # PDF -> Image Converter
│   ├── table_detection.py      # YOLO Table Detection
├── input_pdf/                  # Raw PDF Inputs
├── outputs/                    # Processed Text Outputs
└── requirements.txt            # Python Dependencies
```
