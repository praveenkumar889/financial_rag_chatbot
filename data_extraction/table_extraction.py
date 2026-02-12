import time
import json
import boto3
import os
import sys
import io
import pandas as pd
from botocore.exceptions import ClientError
from dotenv import load_dotenv, find_dotenv
from trp import Document
import re

# Load environment variables
basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(basedir, '.env'), override=True)

# Configuration
ENABLE_NARRATIVE = True
AWS_REGION = os.getenv('AWS_REGION', 'ap-south-1')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
INPUT_PREFIX = os.getenv('INPUT_PREFIX', 'input/')
OUTPUT_PREFIX = os.getenv('OUTPUT_PREFIX', 'output/')
LOCAL_INPUT_DIR = r"C:\financial_agent\input_pdf"
LOCAL_OUTPUT_DIR = r"C:\financial_agent\outputs\table_extraction"
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

print(f"DEBUG: Basedir: {basedir}")
print(f"DEBUG: Loaded S3_BUCKET_NAME: {S3_BUCKET_NAME}")
print(f"DEBUG: Has Access Key: {bool(AWS_ACCESS_KEY_ID)}")

def get_s3_client():
    return boto3.client(
        's3', 
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

def get_textract_client():
    return boto3.client(
        'textract', 
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

def list_input_files(bucket, prefix):
    """List all files in the input S3 folder."""
    s3 = get_s3_client()
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if 'Contents' not in response:
            return []
        # Filter out the folder itself if it appears
        files = [obj['Key'] for obj in response['Contents'] if not obj['Key'].endswith('/')]
        return files
    except ClientError as e:
        print(f"Error accessing S3 bucket {bucket}: {e}")
        return []

def start_table_extraction(bucket, document_key, output_bucket, output_prefix):
    """Starts the asynchronous document analysis for table extraction."""
    textract = get_textract_client()
    
    # Construct the output prefix for this specific file
    # Example: input/file1.pdf -> output/file1.pdf/
    file_name = os.path.basename(document_key)
    specific_output_prefix = f"{output_prefix}{file_name}/"
    
    print(f"Starting extraction for: {document_key}")
    print(f"Textract raw output will be saved to: s3://{output_bucket}/{specific_output_prefix}")

    try:
        response = textract.start_document_analysis(
            DocumentLocation={
                'S3Object': {
                    'Bucket': bucket,
                    'Name': document_key
                }
            },
            FeatureTypes=['TABLES'],
            OutputConfig={
                'S3Bucket': output_bucket,
                'S3Prefix': specific_output_prefix
            }
        )
        return response['JobId']
    except ClientError as e:
        print(f"Error starting Textract job for {document_key}: {e}")
        return None

def get_job_results(job_id):
    """Waits for the job to complete and retrieves the results."""
    textract = get_textract_client()
    print(f"Waiting for job {job_id} to complete...")
    
    while True:
        try:
            response = textract.get_document_analysis(JobId=job_id)
            status = response['JobStatus']
            
            if status == 'SUCCEEDED':
                print(f"Job {job_id} succeeded! Retrieving results...")
                
                # Pagination handling
                blocks = response['Blocks']
                next_token = response.get('NextToken')
                
                while next_token:
                    response = textract.get_document_analysis(JobId=job_id, NextToken=next_token)
                    blocks.extend(response['Blocks'])
                    next_token = response.get('NextToken')
                
                return {'JobStatus': status, 'Blocks': blocks}
                
            elif status == 'FAILED':
                print(f"Job {job_id} failed: {response.get('StatusMessage')}")
                return None
            else:
                print(f"Status: {status}. Waiting...")
                time.sleep(5)
                
        except ClientError as e:
            print(f"Error checking job status: {e}")
            return None

def ensure_bucket_folders(bucket, folders):
    """Ensure specific folders exist in the S3 bucket."""
    s3 = get_s3_client()
    for folder in folders:
        # Ensure folder path ends with /
        if not folder.endswith('/'):
            folder += '/'
            
        try:
            s3.put_object(Bucket=bucket, Key=folder)
            print(f"Ensured folder exists: s3://{bucket}/{folder}")
        except ClientError as e:
            print(f"Error creating folder {folder}: {e}")

def sync_local_to_s3(local_folder, bucket, s3_prefix):
    """Uploads local PDF files to S3 input folder if they don't exist."""
    print(f"Syncing local files from {local_folder} to s3://{bucket}/{s3_prefix}...")
    s3 = get_s3_client()
    
    if not os.path.exists(local_folder):
        print(f"Local input folder {local_folder} does not exist.")
        return

    for filename in os.listdir(local_folder):
        if filename.lower().endswith('.pdf'):
            local_path = os.path.join(local_folder, filename)
            s3_key = f"{s3_prefix}{filename}"
            
            # Check if file already exists in S3 (simple check)
            try:
                s3.head_object(Bucket=bucket, Key=s3_key)
                # print(f"File {filename} already exists in S3. Skipping.")
            except ClientError:
                # File missing, upload it
                print(f"Uploading {filename} to S3...")
                s3.upload_file(local_path, bucket, s3_key)

def infer_financial_metadata(text):
    """Infer financial tags using robust keywords."""
    t = text.lower()
    if any(k in t for k in ["cash flow", "operating activities", "investing activities"]):
        return "cash_flow"
    if any(k in t for k in ["revenue", "profit", "income", "expense"]):
        return "income_statement"
    if any(k in t for k in ["assets", "liabilities", "equity"]):
        return "balance_sheet"
    return "other"

def table_to_narrative(table_data):
    """
    Generates a deterministic narrative from table rows.
    Assumes first column is metric/subject and looks for year-based columns (2024, 2025).
    Checks both Row 0 and Row 1 for years to handle complex headers.
    """
    narrative = []
    if len(table_data) < 2:
        return narrative

    # Step 1: Detect which row contains the years (Header Row)
    # We check row 0 and row 1.
    year_map = {} # {col_index: year_string}
    data_start_row = 1
    
    # Check Row 0
    header_row = table_data[0]
    years_0 = {i: h for i, h in enumerate(header_row) if re.fullmatch(r"(19|20)\d{2}", h.strip())}
    
    if years_0:
        year_map = years_0
        data_start_row = 1
    else:
        # Check Row 1 (if exists)
        if len(table_data) > 1:
            years_1 = {i: h for i, h in enumerate(table_data[1]) if re.fullmatch(r"(19|20)\d{2}", h.strip())}
            if years_1:
                year_map = years_1
                data_start_row = 2
    
    # If no valid year columns found in first 2 rows, skip narrative
    if not year_map:
         return []

    def is_numeric(val):
        """Check if value is a valid financial number (allows () for negatives, commas)."""
        v = val.replace(',', '').replace('(', '').replace(')', '').strip()
        return v.replace('.', '', 1).isdigit()

    # Process rows
    for row in table_data[data_start_row:]:
        if len(row) < 2: 
            continue
            
        metric = row[0].strip()
        metric = metric.replace('\n', ' ')
        
        # Skip if metric is empty or purely numeric
        if not metric or metric.replace('.','').isdigit():
            continue

        # Check for section headers (rows with no numeric values in year columns)
        row_values = []
        for col_idx, year_val in year_map.items():
            if col_idx < len(row):
                val = row[col_idx].strip()
                if val and val not in ['-', ''] and is_numeric(val):
                    row_values.append((val, year_val))
        
        if not row_values:
            continue

        # Construct Sentence
        first_val, first_year = row_values[0]
        sentence = f"{metric} was {first_val} in {first_year}"
        
        if len(row_values) > 1:
            comparisons = [f"{v} in {y}" for v, y in row_values[1:]]
            sentence += f", compared to {', '.join(comparisons)}"
            
        sentence += "."
        narrative.append(sentence)

    return narrative

def parse_textract_to_json(textract_json):
    """
    Parses AWS Textract JSON response using TRP and converts tables into clean JSON structure.
    """
    doc = Document(textract_json)
    tables_data = []

    for page_num, page in enumerate(doc.pages):
        for table_index, table in enumerate(page.tables):
            table_content = []
            
            for row in table.rows:
                row_data = []
                for cell in row.cells:
                    row_data.append(cell.text.strip())
                table_content.append(row_data)

            # Generate embedding-friendly text
            flattened_text = " | ".join(
                [" ; ".join(row) for row in table_content]
            )

            # Metadata for formatting
            table_entry = {
                "content_type": "table",
                "table_number": len(tables_data) + 1,
                "page_number": page_num + 1,
                "row_count": len(table_content),
                "column_count": max(len(r) for r in table_content) if table_content else 0,
                "extraction_method": "aws_textract",
                "embedding_text": flattened_text,
                "narrative_text": (
                    table_to_narrative(table_content)
                    if ENABLE_NARRATIVE and infer_financial_metadata(flattened_text) in [
                        "cash_flow", "income_statement", "balance_sheet"
                    ]
                    else []
                ),
                "financial_section": infer_financial_metadata(flattened_text),
                "parsing_report": {
                    "accuracy": 100.0,
                    "order_on_page": table_index + 1
                },
                "data": table_content
            }
            tables_data.append(table_entry)
    return tables_data

def save_to_s3_as_excel(bucket, file_key, tables_data):
    """Saves the extracted tables as an Excel file to S3."""
    s3 = get_s3_client()
    
    # Create an in-memory Excel file
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for i, table in enumerate(tables_data):
            df = pd.DataFrame(table['data'])
            sheet_name = f"Table_{i+1}_Page_{table['page_number']}"
            # Truncate sheet name to 31 chars if needed
            sheet_name = sheet_name[:31] 
            df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
            
    output.seek(0)
    
    # Construct output key
    original_filename = os.path.basename(file_key)
    file_name_no_ext = os.path.splitext(original_filename)[0]
    output_key = f"{OUTPUT_PREFIX}{file_name_no_ext}_extracted.xlsx"
    
    try:
        s3.put_object(Bucket=bucket, Key=output_key, Body=output)
        print(f"Saved EXCEL output to S3: s3://{bucket}/{output_key}")
    except ClientError as e:
        print(f"Error saving Excel to S3: {e}")

def main():
    if not S3_BUCKET_NAME:
        print("Error: S3_BUCKET_NAME not set in .env")
        return
        
    # Ensure local output directory exists
    if not os.path.exists(LOCAL_OUTPUT_DIR):
        os.makedirs(LOCAL_OUTPUT_DIR)
        print(f"Created local output directory: {LOCAL_OUTPUT_DIR}")

    # Ensure input and output folders exist in the bucket
    ensure_bucket_folders(S3_BUCKET_NAME, [INPUT_PREFIX, OUTPUT_PREFIX])

    # Ensure input and output folders exist in the bucket
    ensure_bucket_folders(S3_BUCKET_NAME, [INPUT_PREFIX, OUTPUT_PREFIX])

    # Sync local files to S3
    sync_local_to_s3(LOCAL_INPUT_DIR, S3_BUCKET_NAME, INPUT_PREFIX)

    print(f"Scanning bucket '{S3_BUCKET_NAME}' folder '{INPUT_PREFIX}'...")
    
    files = list_input_files(S3_BUCKET_NAME, INPUT_PREFIX)
    
    # Filter to only process INFY_Tables.pdf as per current focus
    files = [f for f in files if "INFY_Tables.pdf" in f]
    
    if not files:
        print("No matching files found in the input folder.")
        return

    print(f"Found {len(files)} files. Starting processing...")

    for file_key in files:
        job_id = start_table_extraction(
            bucket=S3_BUCKET_NAME,
            document_key=file_key,
            output_bucket=S3_BUCKET_NAME,
            output_prefix=OUTPUT_PREFIX
        )
        
        if job_id:
            print(f"Job started! Job ID: {job_id}")
            result = get_job_results(job_id)
            
            if result:
                try:
                    # 1. Create Document Metadata
                    document_metadata = {
                        "document_name": os.path.basename(file_key),
                        "s3_bucket": S3_BUCKET_NAME,
                        "s3_key": file_key,
                        "textract_job_id": job_id,
                        "extraction_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "source": "aws_textract"
                    }

                    # 2. Parse into Clean JSON
                    clean_tables = parse_textract_to_json(result)
                    
                    # 3. Attach metadata to each table
                    for table in clean_tables:
                        table["document_metadata"] = document_metadata

                    # 4. Save JSON Locally
                    original_filename = os.path.basename(file_key)
                    file_name_no_ext = os.path.splitext(original_filename)[0]
                    local_filename = f"{file_name_no_ext}_tables.json"
                    local_path = os.path.join(LOCAL_OUTPUT_DIR, local_filename)
                    
                    with open(local_path, 'w') as f:
                        json.dump(clean_tables, f, indent=4)
                    print(f"Saved CLEAN JSON output locally to: {local_path}")

                    # 4b. Save Narrative Semantic Text Locally (For RAG/Embeddings)
                    txt_filename = f"{file_name_no_ext}_tables.txt"
                    txt_path = os.path.join(LOCAL_OUTPUT_DIR, txt_filename)
                    
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        for table in clean_tables:
                            header = f"Table {table['table_number']} | Page {table['page_number']} | Type: {table['financial_section']}"
                            f.write(f"[{header}]\n")
                            f.write("RAW DATA: " + table['embedding_text'] + "\n")
                            if table.get('narrative_text'):
                                f.write("NARRATIVE:\n")
                                for sentence in table['narrative_text']:
                                    f.write(f"- {sentence}\n")
                            f.write("-" * 80 + "\n")
                    print(f"Saved SEMANTIC TEXT output locally to: {txt_path}")
                    
                    # 5. Save JSON to S3 (New)
                    s3 = get_s3_client()
                    json_output_key = f"{OUTPUT_PREFIX}{file_name_no_ext}_tables.json"
                    try:
                        s3.put_object(
                            Bucket=S3_BUCKET_NAME,
                            Key=json_output_key,
                            Body=json.dumps(clean_tables, indent=2)
                        )
                        print(f"Saved JSON output to S3: s3://{S3_BUCKET_NAME}/{json_output_key}")
                    except ClientError as e:
                        print(f"Error saving JSON to S3: {e}")

                    # 6. Save Excel to S3
                    save_to_s3_as_excel(S3_BUCKET_NAME, file_key, clean_tables)
                    
                except Exception as e:
                    print(f"Error processing results: {e}")
            
            print("-" * 50)

if __name__ == "__main__":
    main()
