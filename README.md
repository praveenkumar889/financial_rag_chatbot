# RAG Chatbot Project

This project extracts text from PDF files using AWS Textract and builds a RAG (Retrieval Augmented Generation) chatbot using LangChain and OpenAI.

## Prerequisites

1. Python 3.9+
2. AWS Account with Textract access and an S3 bucket.
3. OpenAI API Key.

## Setup

1. **Install Dependencies**
   Run this from the `financial_agent` root directory:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   Create a `.env` file in the `financial_agent` root directory:
   ```env
   AWS_REGION=us-east-1
   S3_SOURCE_BUCKET_NAME=your-source-bucket-name
   S3_DEST_BUCKET_NAME=your-destination-bucket-name
   OPENAI_API_KEY=your-openai-api-key
   ```

## Usage

### 1. Extract Data
Extract text from your local PDF images. This script uploads the PDF to your S3 bucket, runs Textract, and saves the output JSON locally in `data_extraction`.

```bash
cd data_extraction
python extract_text.py "C:/path/to/your/image.pdf" .
```
This will generate `.json` files in the `data_extraction` folder.

### 2. Run Chatbot
Start the Streamlit chatbot application. It will automatically load any `.json` files found in the `data_extraction` folder.

```bash
cd rag_chatbot
streamlit run rag_app.py
```

## Project Structure
- `data_extraction/`: Scripts for processing PDFs using AWS Textract.
- `rag_chatbot/`: The RAG application source code.
- `requirements.txt`: Python dependencies.
