import fitz  # pymupdf
import os
import re
import sys
import json
import hashlib
from collections import Counter
from pathlib import Path

# -----------------------------------------------------------------------------
# ⚙️ CONFIGURATION & INPUTS
# -----------------------------------------------------------------------------
SELECTED_PDF = r"C:\financial_agent\input_pdf\INFY_2025_transcript.pdf"

# JOBS CONFIGURATION
CONFIGS = [
    {
        "path": SELECTED_PDF,
        "skip_pages": set(), # Process ALL pages for transcripts
        "output_folder": r"C:\financial_agent\outputs\transcript_extraction"
    }
]

class PDFNarrativeExtractor:
    def __init__(self, pdf_path, table_pages_to_skip=None):
        self.pdf_path = pdf_path
        self.doc = None
        self.blocks = []
        self.output_text = [] 
        self.award_blocks = [] 
        self.table_pages_to_skip = table_pages_to_skip or set()
        
        # Doc Type Detection
        self.doc_type = "TRANSCRIPT"
            
        print(f"initialized Config: DocType={self.doc_type}, SkipPages={len(self.table_pages_to_skip)}")
        
        self.seen_hashes = set()
        self.stats = {
            "pages_processed": 0,
            "narrative_pages": 0,
            "table_pages_skipped": 0,
            "blocks_processed": 0,
            "low_confidence_blocks": 0,
            "duplicate_blocks_removed": 0
        }
        self.body_font_size = 11.0 
        
        self.bullet_pattern = re.compile(r'^(\u2022|\u2023|\u25E6|\u2043|\u2219|\u25AA|\*|\-)\s')
        self.number_list_pattern = re.compile(r'^(\d+)\.\s')
        self.page_num_pattern = re.compile(r'^(page\s?\d+|\d+\s?/\s?\d+|\d+)$', re.IGNORECASE)

    def load_pdf(self):
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF not found at {self.pdf_path}")
        self.doc = fitz.open(self.pdf_path)
        print(f"Loaded PDF: {self.pdf_path} ({len(self.doc)} pages)")

    def detect_reading_order(self):
        print("Reconstructing reading order...")
        all_blocks = []
        
        for page_num, page in enumerate(self.doc):
            current_page = page_num + 1
            self.stats["pages_processed"] += 1
            self.stats["narrative_pages"] += 1

            page_dict = page.get_text("dict")
            width = page_dict['width']
            height = page_dict['height']
            blocks = page_dict.get("blocks", [])
            
            header_threshold = height * 0.05
            footer_threshold = height * 0.95
            left_margin_threshold = width * 0.05 
            right_margin_threshold = width * 0.95

            def smart_join_spans(spans):
                texts = [s['text'] for s in spans]
                joined = "".join(texts)
                if len(joined) > 5 and sum(len(t) == 1 for t in texts) / max(len(texts), 1) > 0.6:
                    return joined
                return " ".join(texts)

            for i, block in enumerate(blocks):
                if block['type'] == 0:
                    bbox = block['bbox']
                    if bbox[2] < left_margin_threshold or bbox[0] > right_margin_threshold:
                         continue

                    lines = block.get('lines', [])
                    lines_text = " ".join(smart_join_spans(line['spans']) for line in lines).strip()
                    
                    all_blocks.append({
                        'page': current_page,
                        'bbox': bbox,
                        'lines': lines,
                        'page_width': width,
                        'full_text_preview': lines_text,
                        'is_header_footer': bbox[3] < header_threshold or bbox[1] > footer_threshold,
                        'block_id': f"{current_page}_{i}"
                    })

        # Sort Logic
        for block in all_blocks:
            bbox = block['bbox']
            page_width = block['page_width']
            block_width = bbox[2] - bbox[0]
            if block_width > page_width * 0.7:
                block['column'] = -1
            else:
                mid = page_width / 2
                block_center = (bbox[0] + bbox[2]) / 2
                block['column'] = 0 if block_center < mid else 1
        
        self.blocks = sorted(all_blocks, key=lambda b: (b['page'], b['bbox'][1], b.get('column', 0)))

    def fix_text_issues(self, text):
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1-\2', text)
        text = re.sub(r'\b(customer)centric\b', r'\1-centric', text, flags=re.IGNORECASE)
        text = re.sub(r'(?<![.!?])\n(?!\n)', ' ', text)
        return text

    def detect_section_boundaries(self):
        """DETECT SPEAKER NAMES AS HEADINGS"""
        print("Detecting speaker boundaries...")
        if not self.blocks: return

        # Font Analysis
        font_sizes = []
        for block in self.blocks:
            for line in block.get('lines', []):
                for span in line['spans']:
                    font_sizes.append(round(span['size'], 1))
        
        if not font_sizes: return
        self.body_font_size = Counter(font_sizes).most_common(1)[0][0]
        # For transcripts, speakers are often BOLD but same size, or slightly larger.
        heading_threshold = self.body_font_size * 1.05 

        for block in self.blocks:
            try:
                lines = block.get('lines', [])
                if not lines:
                    block['role'] = 'BODY'
                    continue

                text_sample = " ".join(span['text'] for line in lines for span in line['spans']).strip()
                if not text_sample:
                    block['role'] = 'BODY'
                    continue

                first_span = lines[0]['spans'][0]
                size = first_span['size']
                font_name = first_span.get('font', '').lower()
                
                is_bold = "bold" in font_name
                is_large = size >= heading_threshold
                words = text_sample.split()
                
                # SPEAKER DETECTION LOGIC
                # 1. Bold text usually indicates speaker name in transcripts
                # 2. Short length (2-4 words) e.g., "Rishi Basu", "Salil Parekh"
                # 3. Title Case
                # 4. No verbs (usually)
                
                is_short_name = len(words) <= 4 and len(words) >= 1
                is_title = text_sample.istitle() or all(w[0].isupper() for w in words if w[0].isalpha())
                
                # Check for verbs to avoid short sentences being detected as names
                has_verbs = bool(re.search(r'\b(is|was|are|were|has|have|had|said|says)\b', text_sample.lower()))
                
                # Specific Check: Known structure "Name" on a single line
                is_speaker_candidate = (is_bold or is_short_name) and is_title and not has_verbs
                
                if is_speaker_candidate:
                     block['role'] = 'HEADING' # Treated as Speaker Name
                else:
                    block['role'] = 'BODY'

            except (IndexError, KeyError):
                block['role'] = 'BODY'

    def filter_text_blocks(self):
        # Simplistic filter for transcript
        filtered_blocks = []
        for block in self.blocks:
            text = " ".join(span['text'] for line in block.get('lines', []) for span in line['spans']).strip()
            if text:
                text = self.fix_text_issues(text)
                block['full_text'] = text
                
                # Remove Page Numbers
                if self.page_num_pattern.match(text) or re.match(r'^\d+\s*$', text):
                    continue
                # Remove Footer boilerplate (Infosys specific)
                if "infosys limited" in text.lower() and len(text) < 50:
                    continue

                filtered_blocks.append(block)
        
        self.blocks = filtered_blocks

    def remove_headers_footers(self):
        pass # Simplified for transcript

    def process_content_structure(self):
        print("Processing transcript structure...")
        doc_name = Path(self.pdf_path).stem
        final_sections = []
        
        current_speaker = "Unknown" 
        current_buffer = [] 
        current_pages = set() 
        
        # We treat "Sections" as "Speaker Segments"
        
        for block in self.blocks:
            text = block['full_text'].strip()
            role = block.get('role', 'BODY')
            page = block['page']

            if role == 'HEADING': 
                # START NEW SPEAKER SEGMENT
                if current_buffer:
                    self._flush_section(final_sections, doc_name, current_speaker, current_pages, current_buffer)
                    current_buffer = []
                    current_pages = set()
                
                # Update Speaker
                current_speaker = text
                # We do NOT add the speaker name to the buffer text, it is metadata.
                continue

            # Accumulate Body
            # Add Metadata Tags
            tags = []
            if '%' in text: tags.append('METRIC')
            if re.search(r'[\$₹€£]', text): tags.append('FINANCIAL')
            if re.search(r'\b(award|recognition|winner)\b', text.lower()): tags.append('RECOGNITION')
            if re.search(r'\b(employees|workforce|attrition|headcount)\b', text.lower()): tags.append('HR_METRIC')
            
            tagged_text = text
            if tags:
                 tagged_text = f"[{' | '.join(tags)}] {text}"

            current_buffer.append(tagged_text)
            current_pages.add(page)
        
        # Flush last
        if current_buffer:
             self._flush_section(final_sections, doc_name, current_speaker, current_pages, current_buffer)

        self.output_text = final_sections

    def _flush_section(self, sections, doc_name, speaker, pages, buffer):
        if not buffer: return
        content = "\n".join(buffer)
        
        # Format Participant Lists (Smart Comma Insertion)
        if speaker and ("CORPORATE PARTICIPANTS" in speaker or "JOURNALISTS" in speaker):
            # Split by 2+ spaces or newlines to get distinct tokens
            tokens = re.split(r'\s{2,}|\n', content)
            tokens = [t.strip() for t in tokens if t.strip()]
            
            non_name_keywords = {
                "officer", "director", "head", "president", "manager", "chief", "lead", 
                "times", "profit", "standard", "express", "herald", "tv18", "cnbc", "ndtv", 
                "moneycontrol", "mint", "business", "hindu", "reuters", "fortune", "bureau", 
                "india", "news", "chronicle", "journal", "post", "street", "bloomberg"
            }
            
            reconstructed = []
            current_group = []
            
            for token in tokens:
                # Check if token is likely a Title/Org
                is_title_or_org = any(k in token.lower() for k in non_name_keywords)
                if token.lower().startswith("the "):
                    is_title_or_org = True
                
                if not is_title_or_org:
                    # Likely a Name -> Start new group
                     if current_group:
                         reconstructed.append("    ".join(current_group))
                         current_group = []
                     current_group.append(token)
                else:
                    # Title/Org -> Append to current
                    if not current_group:
                        current_group.append(token)
                    else:
                        current_group.append(token)
            
            if current_group:
                reconstructed.append("    ".join(current_group))
                
            content = ",    ".join(reconstructed)
        
        page_nums = sorted(list(pages)) if pages else [1]
        start_page = min(page_nums)
        end_page = max(page_nums)
        page_span = f"{start_page}-{end_page}" if start_page != end_page else str(start_page)

        sections.append({
            "id": f"{doc_name}_SPKU_{len(sections)}", # SPKU = Speaker Utterance
            "speaker": speaker,
            "content": content,
            "page": start_page,
            "source_pages": page_nums,
            "page_span": page_span,
            "doc_name": doc_name,
            "type": "TRANSCRIPT_SEGMENT"
        })

    def generate_narrative_output(self, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            for sec in self.output_text:
                # Add Metadata Header for RAG ingestion
                # Format: [METADATA | Key=Val | Key2=Val2]
                f.write(f"[METADATA | Page={sec['page_span']} | Speaker={sec['speaker']}]\n")
                
                f.write(f"SPEAKER: {sec['speaker']}\n")
                f.write(f"Pages: {sec['page_span']}\n")
                f.write("-" * 20 + "\n")
                f.write(sec['content'] + "\n")
                f.write("\n" + "="*50 + "\n\n")

    def export_json(self, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.output_text, f, indent=4)

    def print_extraction_stats(self):
        print("Extraction Complete.")

def run_extraction(pdf_path, output_folder=None):
    if output_folder is None:
        output_folder = r"C:\financial_agent\outputs\transcript_extraction"

    input_path = pdf_path
    
    print(f"\n{'='*50}")
    print(f"🚀 STARTING JOB: {str(Path(input_path).name)}")
    print(f"📂 Output Folder: {output_folder}")
    print(f"{'='*50}")
    
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return None

    input_filename = Path(input_path).stem
    output_dir = output_folder
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{input_filename}.txt")
    json_output_file = os.path.join(output_dir, f"{input_filename}.json")

    extractor = PDFNarrativeExtractor(input_path)
    try:
        extractor.load_pdf()
        extractor.detect_reading_order()    
        extractor.detect_section_boundaries() 
        extractor.filter_text_blocks()
        extractor.process_content_structure() 
        extractor.generate_narrative_output(output_file)
        extractor.export_json(json_output_file)
        extractor.print_extraction_stats()
        print(f"\n✅ Success! Output saved to:\n{output_file}")
        if extractor.doc:
            extractor.doc.close()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

    return output_file

if __name__ == "__main__":
    print(f"🎯 Target PDF: {Path(SELECTED_PDF).name}")
    target_config = next((c for c in CONFIGS if os.path.normpath(c["path"]) == os.path.normpath(SELECTED_PDF)), None)
    
    if target_config:
        print("✅ Found matching configuration.")
        run_extraction(target_config["path"], target_config["output_folder"])
    else:
        print("⚠️ No specific config found - Running generic extraction")
        run_extraction(SELECTED_PDF, output_folder=r"C:\financial_agent\outputs\transcript_extraction")
