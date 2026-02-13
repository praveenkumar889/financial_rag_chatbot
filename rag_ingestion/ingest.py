
import os
import json
import pymongo
from pymongo import MongoClient, UpdateOne
from openai import AzureOpenAI
from dotenv import load_dotenv, find_dotenv
import uuid
import hashlib
from datetime import datetime
import re

# Load environment variables
load_dotenv(find_dotenv(filename="../.env"), override=True)

# Azure Configuration for Embeddings
AZURE_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_AI_API_KEY")
# Using the model specified by user
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002") 

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGO_DB_NAME", "financial_rag")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "rag_chunks")

client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version="2024-02-15-preview"
)

def get_embedding(text):
    """Generates embedding using Azure OpenAI."""
    text = text.replace("\n", " ")
    try:
        response = client.embeddings.create(
            input=[text],
            model=EMBEDDING_DEPLOYMENT
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def generate_chunk_id(doc_name, text, company="Unknown", year="Unknown"):
    """Generates a deterministic ID based on document name, metadata, and text content."""
    # Create a unique string based on doc and content
    # Improvement 2: Include Company and Year for collision safety
    unique_string = f"{company}_{year}_{doc_name}_{text}"
    # Return MD5 hash
    return hashlib.md5(unique_string.encode('utf-8')).hexdigest()

import certifi

def connect_mongo():
    try:
        # ROBUST CONNECTION SETTINGS (Fixes SSL Handshake / Idle Timeouts)
        client = MongoClient(
            MONGO_URI,
            tls=True,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=30000,
            socketTimeoutMS=60000, 
            connectTimeoutMS=30000,
            maxPoolSize=50,
            minPoolSize=5,
            retryWrites=True
        )
        
        # Verify connection immediately
        client.admin.command("ping")
        print("MongoDB ping successful")
        
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Create Vector Index (Atlas Search - requires manual setup in Atlas UI, 
        # but locally we just insert vectors).
        # We can create a standard index on metadata fields though.
        collection.create_index([("metadata.document_name", pymongo.ASCENDING)])
        collection.create_index([("metadata.content_type", pymongo.ASCENDING)])
        # Improvement 1: Unique Index on chunk_id
        collection.create_index([("chunk_id", pymongo.ASCENDING)], unique=True)
        
        print(f"Connected to MongoDB: {DB_NAME}.{COLLECTION_NAME}")
        return collection
    except Exception as e:
        print(f"MongoDB Connection Error: {e}")
        return None

def parse_header_metadata(text):
    """Parses [METADATA | Key=Val] header from text blocks."""
    metadata = {}
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        if line.strip().startswith("[METADATA"):
            # Parse: [METADATA | Page=1 | FY=FY2025 ...]
            content = line.strip().strip("[]").replace("METADATA |", "")
            parts = content.split("|")
            for part in parts:
                if "=" in part:
                    k, v = part.split("=", 1)
                    metadata[k.strip().lower().replace(" ", "_")] = v.strip()
        else:
            cleaned_lines.append(line)
            
    return "\n".join(cleaned_lines), metadata

def chunk_narrative_text(chunk_text, source_meta):
    """Chunks narrative text with overlap."""
    chunks = []
    
    # Simple semantic splitting by double newline or headers
    # For narrative files, we split by "###" sections or paragraphs
    
    # First, separate by sections
    sections = re.split(r'\n(?=###|\[Table|\[Chart)', chunk_text)
    
    for section in sections:
        if not section.strip():
            continue
            
        # Refine metadata from section content if possible
        section_meta = source_meta.copy()
        
        # ---------------------------------------------------------
        # SECTION HEADER PARSING (Annual Report / Structured Narrative)
        # [SECTION ID: ... | LEVEL: ... | IMPORTANCE: ...]
        # ---------------------------------------------------------
        header_match = re.match(r'\[SECTION ID:\s*(.*?)\s*\|\s*LEVEL:\s*(.*?)\s*\|\s*IMPORTANCE:\s*([\d.]+)\]', section.strip())
        if header_match:
            section_meta["section_id"] = header_match.group(1)
            section_meta["section_level"] = header_match.group(2)
            try:
                section_meta["importance"] = float(header_match.group(3))
            except:
                section_meta["importance"] = 1.0
            
            # REMOVE ID LINE FROM TEXT (Cleaner embeddings)
            section = section.replace(header_match.group(0), "").strip()

        # Ensure default importance if missing
        section_meta.setdefault("importance", 1.0)
        
        # EXTRACT METRIC TAGS (e.g. [HR_METRIC])
        metric_tags = re.findall(r'\[(HR_METRIC|FINANCIAL|METRIC.*?)\]', section)
        if metric_tags:
            # Normalize tags (remove "METRIC |" prefix)
            cleaned_tags = [t.replace("METRIC |", "").strip() for t in metric_tags]
            section_meta["metric_tags"] = cleaned_tags

        # EXTRACT SECTION TITLE (e.g. ### Financial Highlights)
        if section.strip().startswith("###"):
            title_line = section.strip().split("\n")[0]
            section_meta["section_title"] = title_line.replace("###", "").strip()

        # ---------------------------------------------------------
        # BULLET POINT CHUNKING (Lists in Annual Reports)
        # ---------------------------------------------------------
        # Check if section is primarily a list (contains bullets at start of lines)
        source_is_report = source_meta.get("document_type") == "annual_report"
        has_bullets = re.search(r'^\s*[•\-]', section, re.MULTILINE)
        
        if source_is_report and has_bullets:
             # Extract lines that look like bullets
             lines = section.split('\n')
             bullets = [b.strip() for b in lines if b.strip().startswith(('•', '-'))]
             
             # Also keep intro text if it exists (lines before first bullet)
             intro_text = []
             for line in lines:
                 if line.strip().startswith(('•', '-')):
                     break
                 if line.strip():
                     intro_text.append(line.strip())
             
             # 1. Chunk Intro Text (if significant)
             if intro_text:
                 intro_blob = " ".join(intro_text)
                 if len(intro_blob) > 50:
                     chunks.append({
                        "text": intro_blob,
                        "metadata": {**section_meta, "content_type": "section_intro"}
                    })

             # 2. Chunk Bullets (Group of 5)
             buffer = []
             for b in bullets:
                 buffer.append(b)
                 if len(buffer) >= 5:
                     chunks.append({
                         "text": "\n".join(buffer),
                         "metadata": {**section_meta, "content_type": "bullet_points"}
                     })
                     buffer = []
             
             if buffer:
                 chunks.append({
                     "text": "\n".join(buffer),
                     "metadata": {**section_meta, "content_type": "bullet_points"}
                 })
             
             continue # Skip downstream logic for this section
        
        # Check if table/chart header
        if section.strip().startswith("[Table"):
             # Extract table info: [Table 1 | Page 1 | Type: cash_flow]
             headers = re.match(r'\[Table (\d+) \| Page (\d+) \| Type: (\w+)\]', section)
             if headers:
                 section_meta["table_number"] = headers.group(1)
                 section_meta["page_number"] = headers.group(2)
                 section_meta["financial_section"] = headers.group(3)
                 section_meta["content_type"] = "table_narrative"

        # ---------------------------------------------------------
        # PPT / ANALYSIS FILES (Investor decks, summaries, charts)
        # ---------------------------------------------------------
        if source_meta.get("document_type") == "ppt":
            paragraphs = [p.strip() for p in section.split("\n\n") if p.strip()]

            buffer = []
            max_chars = 900

            for p in paragraphs:
                buffer.append(p)
                if len("\n\n".join(buffer)) >= max_chars:
                    chunks.append({
                        "text": "\n\n".join(buffer),
                        "metadata": {
                            **section_meta,
                            "content_type": "ppt_analysis"
                        }
                    })
                    buffer = []

            if buffer:
                chunks.append({
                    "text": "\n\n".join(buffer),
                    "metadata": {
                        **section_meta,
                        "content_type": "ppt_analysis"
                    }
                })

            continue

        # ---------------------------------------------------------
        # TABLE CHUNKING (Raw Data vs Narrative split)
        # ---------------------------------------------------------
        if source_meta.get("document_type") == "table":
             raw_data_text = ""
             narrative_text = ""
             
             # Split section into RAW and NARRATIVE parts
             if "RAW DATA:" in section:
                 if "NARRATIVE:" in section:
                     parts = section.split("NARRATIVE:")
                     raw_part = parts[0]
                     narrative_text = parts[1]
                 else:
                     raw_part = section
                     narrative_text = ""
                 
                 # Clean "RAW DATA:" label
                 idx = raw_part.find("RAW DATA:")
                 if idx != -1:
                     raw_data_text = raw_part[idx + len("RAW DATA:"):].strip()
            
             # 2. Process RAW DATA
             if raw_data_text:
                 # SPLIT ON PIPE '|' (Correct logical delimiter for this file format)
                 # Do NOT normalize to newlines first
                 rows = [r.strip() for r in raw_data_text.split('|') if r.strip()]
                 
                 # Financial row filtering (skip headers/section titles)
                 SKIP_PREFIXES = (
                     "Particulars", "Year ended", "As at March", "Numerator",
                     "Cash flow from", "Adjustments to reconcile", 
                     "Changes in assets", "Cash flows from", 
                     "Payments to acquire", "Proceeds on sale",
                     "Supplementary information", "Items that will",
                     "(In ", "Equity and liabilities", "Assets", "Liabilities"
                 )
                 
                 # Chunking Strategy: 1 row = 1 chunk (High Precision)
                 for row in rows:
                     # Filter out non-data rows
                     if row.startswith(SKIP_PREFIXES) or row == ";" or row == " ; " or not row:
                         continue

                     # Extract row name (first column) for metadata
                     row_base = row.split(';')[0].strip()
                     
                     chunks.append({
                         "text": row,
                         "metadata": {
                             "row_name": row_base, # Add explicit row key
                             **section_meta, 
                             "content_type": "table_row"
                         }
                     })
            
             # 3. Process NARRATIVE
             if narrative_text:
                 bullets = [line.strip() for line in narrative_text.split('\n') if line.strip()]
                 
                 # Chunking Strategy: Group 3-6 bullets (Semantic Cohesion)
                 buffer = []
                 for b in bullets:
                     buffer.append(b)
                     if len(buffer) >= 5:
                         chunks.append({
                             "text": "\n".join(buffer),
                             "metadata": {**section_meta, "content_type": "table_narrative"}
                         })
                         buffer = []
                 if buffer:
                     chunks.append({
                         "text": "\n".join(buffer),
                         "metadata": {**section_meta, "content_type": "table_narrative"}
                     })
             
             continue
        # ---------------------------------------------------------
        
        # ---------------------------------------------------------
        # TRANSCRIPT CHUNKING (Speaker-based with Role Detection)
        # ---------------------------------------------------------
        if source_meta.get("document_type") == "transcript":
            # Split by SPEAKER: label
            speaker_blocks = re.split(r'\n(?=SPEAKER:)', section)

            for block in speaker_blocks:
                if not block.strip():
                    continue

                lines = block.strip().split('\n')
                
                # Check if this block actually starts with a SPEAKER: tag (re.split keeps the lookahead match)
                if lines[0].startswith("SPEAKER:"):
                    speaker_line = lines[0]
                    speaker_name = speaker_line.replace("SPEAKER:", "").strip()
                    
                    # Heuristic for Role Detection
                    role = "answer"
                    if any(x in speaker_name.lower() for x in ["ritu singh", "journalist", "reporter", "analyst", "moderator"]):
                         role = "question"
                         
                    # Remaining lines are the speech
                    speech_text = " ".join(lines[1:]).strip()
                else:
                    # Fallback for blocks without explicit speaker tag (continuation or intro)
                    speaker_name = "Unknown"
                    role = "narrative"
                    speech_text = block.strip()

                if len(speech_text) < 50: # Skip very short utterances
                    continue
                
                # Further split speech_text if it's too long (e.g. CEO monologue)
                max_chars = 900
                if len(speech_text) > max_chars:
                    sub_sentences = re.split(r'(?<=[.!?])\s+', speech_text)
                    buffer = []
                    for s in sub_sentences:
                        buffer.append(s)
                        if len(" ".join(buffer)) >= max_chars:
                             chunks.append({
                                "text": " ".join(buffer),
                                "metadata": {
                                    **section_meta,
                                    "content_type": "transcript_speech",
                                    "speaker": speaker_name,
                                    "role": role
                                }
                            })
                             buffer = []
                    if buffer:
                         chunks.append({
                            "text": " ".join(buffer),
                            "metadata": {
                                **section_meta,
                                "content_type": "transcript_speech",
                                "speaker": speaker_name,
                                "role": role
                            }
                        })
                else:
                    # Single chunk
                    chunks.append({
                        "text": speech_text,
                        "metadata": {
                            **section_meta,
                            "content_type": "transcript_speech",
                            "speaker": speaker_name,
                            "role": role
                        }
                    })

            continue
        # ---------------------------------------------------------

        # Default narrative chunking (Sentence-based)
        # Using regex to split by sentence delimiters (. ? !)
        sentences = re.split(r'(?<=[.!?])\s+', section)
        
        buffer = []
        max_chars_narrative = 900
        
        for s in sentences:
            s = s.strip()
            if not s: continue
            
            buffer.append(s)
            
            # Check size of current buffer
            current_text = " ".join(buffer)
            
            if len(current_text) >= max_chars_narrative:
                chunks.append({
                    "text": current_text,
                    "metadata": {
                        **section_meta,
                        "content_type": "narrative" # Explicit type
                    }
                })
                buffer = []
                
        # Flush remaining buffer
        if buffer:
            final_text = " ".join(buffer)
            if len(final_text) >= 50: # Skip tiny fragments
                chunks.append({
                    "text": final_text,
                    "metadata": {
                        **section_meta,
                        "content_type": "narrative"
                    }
                })
            
    return chunks

def process_file(filepath, doc_type, collection, additional_metadata=None):
    print(f"Processing {filepath}...")
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by our standardized Metadata Headers or Analysis Headers
    # Pattern: [METADATA | Key=Val] or --- Analysis of Page X ---
    # We will split by newline to process line-by-line streaming style, 
    # but for block-based, let's keep it simple: split by double newline and check for header.
    
    # Improved Strategy:
    # 1. Read entire content.
    # 2. Split by `[METADATA` tags to separate sections.
    
    sections = re.split(r'(?=\[METADATA)', content)
    
    doc_name = os.path.basename(filepath).replace(".txt", "")
    
    base_metadata = {
        "document_name": doc_name,
        "document_type": doc_type,
        "company": additional_metadata.get("company", "Unknown") if additional_metadata else "Unknown",
        "ingested_at": datetime.utcnow()
    }

    if additional_metadata:
        base_metadata.update(additional_metadata)

    chunks_to_insert = []
    BATCH_SIZE = 100 
    total_upserted = 0
    
    for section_content in sections:
        if not section_content.strip():
            continue
            
        # Extract embedded metadata
        cleaned_text, section_meta = parse_header_metadata(section_content)
        
        # Merge metadata
        current_meta = {**base_metadata, **section_meta}
        
        # Normalize page number
        if "page" in current_meta and "page_number" not in current_meta:
            current_meta["page_number"] = current_meta["page"]
        
        # Determine strict content type based on source
        if doc_type == "transcript":
             current_meta["document_type"] = "transcript"
        elif doc_type == "financial_report":
             current_meta["document_type"] = "annual_report"
        elif doc_type == "presentation":
             current_meta["document_type"] = "ppt"
             
        # Chunk the section content
        text_chunks = chunk_narrative_text(cleaned_text, current_meta)
        
        for item in text_chunks:
            # Deterministic ID for Upsert
            # Extract basic metadata for ID generation
            company = current_meta.get("company", "Unknown")
            # Handle variations of 'year' or 'fiscal_year'
            year = current_meta.get("year") or \
                   current_meta.get("fiscal_year") or \
                   current_meta.get("fy") or \
                   "Unknown"

            # ENHANCED: For tables, include page/table context to prevent collisions
            if item["metadata"].get("content_type") == "table_row":
                unique_str = f"{item['metadata'].get('table_number')}_{item['metadata'].get('page_number')}_{item['text']}"
                chunk_id = generate_chunk_id(doc_name, unique_str, company, year)
            else:
                chunk_id = generate_chunk_id(doc_name, item["text"], company, year)
            
            # Improvement 3: Check if embedding exists
            embedding = None
            if collection is not None:
                existing_doc = collection.find_one({"chunk_id": chunk_id}, {"embedding": 1})
                if existing_doc and "embedding" in existing_doc:
                    # Reuse existing embedding
                    embedding = existing_doc["embedding"]
                    # Optionally log skipping: print(f"Skipping embedding for chunk {chunk_id[:8]}...")
            
            if not embedding:
                embedding = get_embedding(item["text"])
            
            if embedding:
                chunk_doc = {
                    "chunk_id": chunk_id,
                    "text": item["text"],
                    "embedding": embedding,
                    "metadata": item["metadata"]
                }
                chunks_to_insert.append(chunk_doc)
                
                # BATCH FLUSH (Prevent timeouts)
                if len(chunks_to_insert) >= BATCH_SIZE and collection is not None:
                     try:
                        operations = [
                            UpdateOne(
                                {"chunk_id": doc["chunk_id"]}, 
                                {"$set": doc},                 
                                upsert=True                    
                            ) for doc in chunks_to_insert
                        ]
                        result = collection.bulk_write(operations)
                        total_upserted += result.upserted_count + result.modified_count
                        print(f"  -> Flushed {len(chunks_to_insert)} chunks...")
                        chunks_to_insert = [] # Reset buffer
                     except Exception as e:
                        print(f"Error flushing batch: {e}")
                
    # Final flush for remaining chunks
    if chunks_to_insert and collection is not None:
        try:
            operations = [
                UpdateOne(
                    {"chunk_id": doc["chunk_id"]}, 
                    {"$set": doc},                 
                    upsert=True                    
                ) for doc in chunks_to_insert
            ]
            
            result = collection.bulk_write(operations)
            total_upserted += result.upserted_count + result.modified_count
            print(f"Processed {doc_name}: {total_upserted} chunks upserted/updated total.")
            
        except Exception as e:
            print(f"Error upserting final chunks: {e}")

def scan_directory_for_txt(directory):
    """Returns a list of full paths to .txt files in the directory."""
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return []
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(".txt")]

def main():
    collection = connect_mongo()
    if collection is None:
        raise Exception("MongoDB connection failed. Aborting pipeline.")
    
    # Dynamic scanning config: subfolder -> document_type
    directories_to_scan = {
        "ppt_extraction": "ppt",
        "table_extraction": "table",
        "transcript_extraction": "transcript",
        "financial_extraction": "annual_report"
    }
    
    base_output_dir = r"C:\financial_agent\outputs"
    
    for subfolder, doc_type in directories_to_scan.items():
        full_dir_path = os.path.join(base_output_dir, subfolder)
        print(f"Scanning {full_dir_path} for .txt files...")
        
        txt_files = scan_directory_for_txt(full_dir_path)
        
        for txt_file in txt_files:
            process_file(txt_file, doc_type, collection)

def run_ingestion(file_path, company, year, quarter, doc_type):
    """
    Orchestrator entry point for single file ingestion.
    """
    collection = connect_mongo()
    if collection is None:
        raise Exception("MongoDB connection failed")

    meta = {
        "company": company,
        "year": year,
        "quarter": quarter
    }
    
    process_file(file_path, doc_type.lower(), collection, additional_metadata=meta)

if __name__ == "__main__":
    main()
