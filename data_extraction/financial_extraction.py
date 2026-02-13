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

# -----------------------------------------------------------------------------
# 🎯 SELECT INPUT FILE
# -----------------------------------------------------------------------------
SELECTED_PDF = r"C:\financial_agent\input_pdf\INFY_FY2025.pdf"

# JOBS CONFIGURATION (Financial Reports Only)
# Note: 'skip_pages' should ideally be loaded from an external config or detected dynamically.
CONFIGS = [
    {
        "path": SELECTED_PDF,
        "skip_pages": set(), # Dynamic loading recommended
        "output_folder": r"C:\financial_agent\outputs\financial_extraction"
    }
]

class PDFNarrativeExtractor:
    def __init__(self, pdf_path, table_pages_to_skip=None):
        self.pdf_path = pdf_path
        self.doc = None
        self.blocks = []
        self.output_text = [] 
        self.award_blocks = [] # New: Structured Fact Store
        self.table_pages_to_skip = table_pages_to_skip or set()
        
        # Doc Type Detection (Strictly Financial)
        self.doc_type = "FINANCIAL_REPORT"
            
        print(f"initialized Config: DocType={self.doc_type}, SkipPages={len(self.table_pages_to_skip)}")
        
        # Production Stats & Quality Metrics
        
        # Production Stats & Quality Metrics
        self.seen_hashes = set()
        self.stats = {
            "pages_processed": 0,
            "narrative_pages": 0,
            "table_pages_skipped": 0,
            "blocks_processed": 0,
            "low_confidence_blocks": 0,
            "duplicate_blocks_removed": 0
        }
        self.body_font_size = 11.0 # Default fallback
        
        # Regex Patterns
        # 🔥 UPGRADE 6: Enhanced Bullet Detection (Added square bullets ▪)
        self.bullet_pattern = re.compile(r'^(\u2022|\u2023|\u25E6|\u2043|\u2219|\u25AA|\*|\-)\s')
        self.number_list_pattern = re.compile(r'^(\d+)\.\s')
        self.page_num_pattern = re.compile(r'^(page\s?\d+|\d+\s?/\s?\d+|\d+)$', re.IGNORECASE)
        self.floating_number_pattern = re.compile(r'^[\d\.\,\%\$₹€£]+$')
        self.metadata_pattern = re.compile(r'\[p=(\d+)\]$')

    def load_pdf(self):
        """Step 1: PDF Layout Detection (PyMuPDF)"""
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF not found at {self.pdf_path}")
        self.doc = fitz.open(self.pdf_path)
        print(f"Loaded PDF: {self.pdf_path} ({len(self.doc)} pages)")

    def detect_reading_order(self):
        """Step 2: Reading Order Reconstruction (Layout Analysis)"""
        print("Reconstructing reading order...")
        all_blocks = []
        
        for page_num, page in enumerate(self.doc):
            current_page = page_num + 1

            # 🧠 RULE ENGINE
            # Only skip table pages if it's a financial report AND pages are provided
            if self.doc_type == "FINANCIAL_REPORT" and current_page in self.table_pages_to_skip:
                print(f"Skipping Page {current_page} (Table Page — handled by Textract)")
                self.stats["table_pages_skipped"] += 1
                continue
            
            self.stats["pages_processed"] += 1
            self.stats["narrative_pages"] += 1

            # Extract raw layout data
            page_dict = page.get_text("dict")
            width = page_dict['width']
            height = page_dict['height']
            blocks = page_dict.get("blocks", [])
            
            # Header/Footer Zones (5% top/bottom)
            header_threshold = height * 0.05
            footer_threshold = height * 0.95
            
            # 1️⃣ PROMPT FIX 1: Ignore blocks in extreme left/right margins (headers/footers)
            left_margin_threshold = width * 0.10 # Increased to 10% to be safe for side notes
            right_margin_threshold = width * 0.90

            # Helper: Fix spaced-out text
            def smart_join_spans(spans):
                texts = [s['text'] for s in spans]
                joined = "".join(texts)
                # If too many single characters -> it's stylized text
                if len(joined) > 5 and sum(len(t) == 1 for t in texts) / max(len(texts), 1) > 0.6:
                    return joined  # No spaces
                return " ".join(texts)

            for i, block in enumerate(blocks):
                if block['type'] == 0:  # Text block
                    bbox = block['bbox']
                    
                    # Margin Filter
                    if bbox[2] < left_margin_threshold or bbox[0] > right_margin_threshold:
                         continue

                    lines = block.get('lines', [])
                    
                    # ✅ FIX: Intelligent Span Join
                    lines_text = " ".join(
                        smart_join_spans(line['spans']) for line in lines
                    ).strip()
                    
                    all_blocks.append({
                        'page': current_page,
                        'bbox': bbox,
                        'lines': lines,
                        # 🔥 FIX 1: Store page width dynamically (User Fix 1)
                        'page_width': width,
                        'full_text_preview': lines_text,
                        'is_header_footer': bbox[3] < header_threshold or bbox[1] > footer_threshold,
                        'block_id': f"{current_page}_{i}"
                    })

        # 1️⃣ PROMPT FIX 1: Sort by Page -> Column -> Y position
        # "Result: Natural human reading flow."
        # 🔥 FIX 1: Column-aware sorting for multi-column layouts
        
        # Assign columns to each block
        for block in all_blocks:
            bbox = block['bbox']
            # 🔥 FIX 1: Use dynamic page width
            page_width = block['page_width']
            
            # 🔥 UPGRADE 1: Full Width Block Detection (Fixes Section Headers)
            block_width = bbox[2] - bbox[0]
            if block_width > page_width * 0.7:
                block['column'] = -1   # FULL WIDTH → read before columns
            else:
                # 🔥 HARDENING 1: Center-based Column Logic (Fixes wide blocks/callouts)
                mid = page_width / 2
                block_center = (bbox[0] + bbox[2]) / 2
                block['column'] = 0 if block_center < mid else 1
        
        # 🔥 ENTRYPRISE PROMPT FIX: Group by Page -> Split Columns -> Sort -> Merge
        # This guarantees correct reading order: Header (-1) -> Left (0) -> Right (1)
        pages = {}
        for b in all_blocks:
            pages.setdefault(b['page'], []).append(b)

        ordered_blocks = []

        for page in sorted(pages.keys()):
            page_blocks = pages[page]
            
            if not page_blocks:
                continue

            # 🔥 ENTERPRISE ADDITION: Auto-detect Single vs Multi-Column
            # Logic: In 2-column, block centers will be far apart (Left=25%, Right=75% -> Spread ~50%)
            # In 1-column, block centers are all roughly middle -> Spread ~ small
            x_positions = [(b['bbox'][0] + b['bbox'][2]) / 2 for b in page_blocks]
            spread = max(x_positions) - min(x_positions)
            page_width = page_blocks[0]['page_width']
            
            # 🔥 FIX: Lower threshold for Annual Reports (30% spread implies columns)
            is_multi_column = spread > (page_width * 0.3)
            
            if not is_multi_column:
                # Single column -> Simple Y-sort
                # 🔥 FIX: Force column ID to 0 to prevent merge errors later
                for b in page_blocks:
                    b['column'] = 0

                ordered_blocks.extend(sorted(page_blocks, key=lambda b: b['bbox'][1]))
                continue

            # Multi-column Logic (Split & Merge)
            # 1. Full Width (Headers/Footers) - Priority 1
            full = [b for b in page_blocks if b.get('column') == -1]
            
            # 2. Left Column - Priority 2
            left = [b for b in page_blocks if b.get('column') == 0]
            
            # 3. Right Column - Priority 3
            right = [b for b in page_blocks if b.get('column') == 1]
            
            # Sort each group by Y position
            full_sorted = sorted(full, key=lambda b: b['bbox'][1])
            left_sorted = sorted(left, key=lambda b: b['bbox'][1])
            right_sorted = sorted(right, key=lambda b: b['bbox'][1])
            
            ordered_blocks.extend(full_sorted + left_sorted + right_sorted)

        self.blocks = ordered_blocks

    def fix_text_issues(self, text):
        """Layer 2: Fix Line-Wrap & Hyphen Splits"""
        
        # 8️⃣ PROMPT FIX 8: Fix Broken Words (Regex Normalization)
        # Issues: "endto-end", "customercentric"
        
        # Pattern 1: Missing Hyphen in Compounds (endto-end -> end-to-end)
        # Heuristic: <word><hyphen><space><word> should be <word>-<word> if likely compound
        # But user example "endto-end" implies "end" + "to" + "-" + "end"? 
        # Actually user wrote: "endto-end". This looks like "end" attached to "to-end".
        # Or maybe "endto - end".
        # Let's handle the specific examples + general heuristic.
        
        # Case A: "word- word" -> "word-word" (Standard line break fix)
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1-\2', text)
        
        # Case B: "customercentric" -> "customer-centric"
        # Hard to do generically without dict, but we can target specific knowns or patterns
        text = re.sub(r'\b(customer)centric\b', r'\1-centric', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(end)to-end\b', r'\1-to-end', text, flags=re.IGNORECASE) # specific fix for user report
        
        # Remove newline inside sentences
        text = re.sub(r'(?<![.!?])\n(?!\n)', ' ', text)
        return text

    def detect_section_boundaries(self):
        """Step 3: Section Boundary Detection (Rule-based NLP)"""
        print("Detecting section boundaries...")
        
        if not self.blocks: return

        # 1. Analyze Document Font Stats
        font_sizes = []
        for block in self.blocks:
            for line in block.get('lines', []):
                for span in line['spans']:
                    font_sizes.append(round(span['size'], 1))
        
        if not font_sizes: return

        # Most common size = Body text
        self.body_font_size = Counter(font_sizes).most_common(1)[0][0]
        heading_threshold = self.body_font_size * 1.15 # 15% larger is usually a heading
        
        print(f"Detected Body Font Size: {self.body_font_size}")
        print(f"Heading Threshold: {heading_threshold}")

        # 2. Tag Blocks as HEADING or BODY
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

                # Analyze first span for style
                first_span = lines[0]['spans'][0]
                size = first_span['size']
                font_name = first_span.get('font', '').lower()
                
                is_bold = "bold" in font_name
                is_large = size >= heading_threshold
                is_short = len(text_sample.split()) <= 10
                is_caps = text_sample.isupper() 
                is_numeric_heavy = (sum(c.isdigit() for c in text_sample) / max(len(text_sample), 1)) > 0.4
                has_punct = any(p in text_sample for p in ".:,;")

                words = text_sample.split()
                is_title_case = len(words) > 0 and all(w[:1].isupper() for w in words if w[:1].isalpha())

                # Heading Heuristics
                is_potential_heading = (is_large or is_bold or is_caps or is_title_case)
                
                # ✅ FIX: Reject signature-like or garbage headings
                if re.fullmatch(r'(sd/|-|_)+', text_sample.lower()) or len(text_sample) < 3:
                    block['role'] = 'BODY'
                    continue

                # Rule: Short, Potential Heading Style, Not Numeric Junk, No Punctuation (usually)
                
                # User Problem 2: Improved Heading Detection
                # Criteria: Short & Title Case OR High Capital Intensity
                u_title = len(words) < 10 and text_sample.istitle()
                u_caps = len(text_sample) > 0 and (sum(1 for c in text_sample if c.isupper()) / len(text_sample) > 0.6)
                
                # --- Fix 2: Signature / Name Block Detector ---
                # Prevent "Name, Title" from being valid headings
                is_signature = bool(re.search(r',\s*(Chief|CIO|CEO|Officer|Head|Director|SVP|EVP|President)', text_sample, re.IGNORECASE))
                starts_with_dash = text_sample.strip().startswith(('–', '-'))
                
                # 🔥 FIX 4: Block job titles without commas (e.g., "Andrew Groth Industry Head – Banking")
                is_title_line = bool(re.search(r'\b(Head|Officer|Director|SVP|EVP|President|Manager)\b', text_sample, re.IGNORECASE)) and len(words) < 8

                # --- Fix 3: Bullet Heading Protection ---
                is_bullet_heading = text_sample.strip().startswith(('-', '•', '·'))
                
                # --- Fix 4: Map / Country Label Filter ---
                # Large isolated words like "Germany", "USA"
                is_geo_label = len(words) <= 3 and text_sample.istitle() and size > self.body_font_size * 1.4

                if is_signature or starts_with_dash or is_bullet_heading or is_geo_label or is_title_line:
                    block['role'] = 'BODY'
                elif (is_short and not has_punct and not is_numeric_heavy and is_potential_heading) or u_title or u_caps:
                     block['role'] = 'HEADING'
                else:
                    block['role'] = 'BODY'
            except (IndexError, KeyError):
                block['role'] = 'BODY'

    def filter_text_blocks(self):
        """Step 4: Text Block Filtering (Noise & Table Removal)"""
        print("Filtering text blocks...")
        filtered_blocks = []
        
        for block in self.blocks:
            self.stats["blocks_processed"] += 1
            
            text = " ".join(
                span['text'] for line in block.get('lines', []) for span in line['spans']
            ).strip()
            
            if text and any(c.isalnum() for c in text):
                # --- 4a. Duplicate Detection (Boilerplate Removal) ---
                text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                if text_hash in self.seen_hashes:
                    self.stats["duplicate_blocks_removed"] += 1
                    continue
                self.seen_hashes.add(text_hash)

                # --- 4b. Confidence Scoring ---
                confidence = 1.0
                words = text.split()
                
                # Penalize small text (often footnotes/artifacts)
                # We need to approximate block font size from lines if not available, 
                # but let's assume valid blocks mostly match valid sizes.
                # For now, simplistic check if we tracked font sizes (complex to get per-block here easily without loop).
                # skipping font check to save perf, relying on text heuristics.
                
                if len(words) < 3: confidence -= 0.2
                if not text[0].isupper() and not text[0].isdigit(): confidence -= 0.1
                # rudimentary encoding check
                if any(ord(c) > 127 for c in text[:10]): confidence -= 0.05 

                block["confidence"] = max(0.0, round(confidence, 2))
                if confidence < 0.7:
                     self.stats["low_confidence_blocks"] += 1

                # --- HYBRID ARCHITECTURE: Block-Level Table Detection ---
                # We do NOT skip entire pages. We skip specific blocks that look like tables.
                
                words = text.split()

                # 🔥 NEW: Extract Awards to Structured Metadata (Fact Store)
                text_lower = text.lower()
                
                # Fix 4: Award Entity Protection
                # Explicitly prevent financial terms from being treated as awards
                if re.search(r'\brevenue recognized|expense recognized|loss recognized\b', text_lower):
                     # Loop continue instead of just flag to ensure we skip award logic entirely but keep flow
                     # But we still want it in the narrative? Yes, just not as an AWARD in metadata.
                     # So we just fall through to normal processing.
                     pass 
                else:
                    # Proceed with award detection checks
                    pass # Just flow control, logic continues below

                
                award_keywords = [
                    "award", "awarded", "winner", "won",
                    "recognized as", "ranked", "top employer",
                    "best workplace", "most ethical", "leader in",
                    "partner of the year", "excellence award"
                ]

                accounting_context = [
                    "recognized revenue", "revenue recognized",
                    "expense recognized", "loss recognized",
                    "impairment", "provision", "fair value",
                    "financial", "balance sheet", "profit or loss"
                ]

                # Must contain award-style phrase
                is_award_style = any(k in text_lower for k in award_keywords)

                # Must NOT contain accounting/reporting language
                is_financial_context = any(k in text_lower for k in accounting_context) or \
                                       bool(re.search(r'\brevenue recognized|expense recognized|loss recognized\b', text_lower))

                # 🔥 FIX 5: Award length adjustment - many awards are longer
                # Length constraint (awards are factual statements, not full paragraphs)
                # FIX 5: Safer Award Logic (Proper Noun Density instead of period check)
                is_facty = len(words) < 50 and sum(1 for w in words if w.istitle()) > 2
                
                # Validation: Awards usually cite the source (by/at/from)
                has_source = bool(re.search(r'\b(by|at|from)\b', text_lower))
                
                # 🔥 HARDENING 4: Strict Financial Guard
                # Exclude purely financial statements even if they look like awards
                if re.search(r'\brevenue|profit|loss|income|expense\b', text_lower):
                    is_award_style = False

                if is_award_style and not is_financial_context and is_facty and has_source:
                    self.award_blocks.append({
                        "page": block["page"],
                        "text": text,
                        "section": "Awards & Recognitions"
                    })
                    continue  # Route to Metadata Store (Skip Narrative)

                numeric_tokens = sum(1 for w in words if re.fullmatch(r'[\d\.,%\$₹€£\-()]+', w))
                numeric_ratio = numeric_tokens / max(len(words), 1)

                # Semantic Check: Sentences usually have verbs. Tables usually don't.
                has_verbs = bool(re.search(r'\b(is|was|are|were|has|have|had|grew|increased|decreased|reported|generated)\b', text.lower()))
                
                # Structure Check: Visual columns or short disjointed text
                many_short_cells = len(words) <= 8
                looks_like_row = "|" in text or "  " in text

                # Rule 1: High Numeric Density + No Verbs -> Likely Data Row
                if numeric_ratio > 0.85 and not has_verbs and many_short_cells:
                    continue # Skip
                
                # Rule 2: Pure Numeric Garbage
                if re.fullmatch(r'[\d\.,%\s\-()₹€£]+', text) and len(words) < 6:
                    continue # Skip

                # Rule 3: Common Table Headers (Particulars | Details | Amount)
                if re.search(r'\b(particulars|description|sl\s*no|sr\s*no)\b.*\b(details|amount|rupees|inr)\b', text.lower()):
                     continue

                # Rule 4: Non-Narrative Fragments (Table Cells)
                if block.get('role') == 'BODY' and len(words) < 10 and not has_verbs:
                    continue
                
                # --- Fix 7: Table Artifact Killer ---
                if len(words) <= 4 and sum(c.isdigit() for c in text) > 0:
                    continue
                
                # --- 4️⃣ PROMPT FIX 4: Remove Non-Knowledge Content ---
                # Delete blocks containing: 
                # TOC listings, Page numbers, Signature lines, URLs, Legal footers, “Scan here”
                
                # TOC Pattern: "Introduction....... 3"
                if re.search(r'\.{3,}\s*\d+', text):
                    continue
                
                # Page Numbers
                if self.page_num_pattern.match(text) or re.match(r'^\d+\s*$', text):
                    continue
                    
                # Signature Lines / Legal Footers
                if re.search(r'^\s*(sd/-|signed by|for and on behalf of)', text, re.IGNORECASE):
                    continue
                
                # URLs
                if "http:" in text or "https:" in text or "www." in text:
                   # If block is mostly URL
                   if len(text.split()) < 5: continue
                
                # QR Scan
                if "scan here" in text.lower() or "scan the qr" in text.lower():
                    continue

                # Apply Layer 2 Fixes
                text = self.fix_text_issues(text)
                block['full_text'] = text
                filtered_blocks.append(block)
        
        # Optimization: deep cleaning to save memory
        for block in filtered_blocks:
            # 🔥 FIX 1: Preserve Geometry for Vertical Merging
            block['y_top'] = block['bbox'][1]
            block['y_bottom'] = block['bbox'][3]
            block.pop('lines', None)
            block.pop('full_text_preview', None)

        self.blocks = filtered_blocks

    def remove_headers_footers(self):
        """Step 5: Header/Footer Removal (Frequency Analysis)"""
        print("Removing headers and footers...")
        
        # 1. Collect candidates from header/footer zones
        header_candidates = []
        for b in self.blocks:
             if b['is_header_footer']:
                 header_candidates.append(b.get('full_text', ''))

        # 2. Identify frequent patterns (appearing > 3 times)
        candidate_counts = Counter(header_candidates)
        
        clean_blocks = []
        report_header_pattern = re.compile(r'Infosys.*Annual Report.*\d+$', re.IGNORECASE)

        for b in self.blocks:
            txt = b.get('full_text', '')
            
            is_freq_header = b['is_header_footer'] and candidate_counts.get(txt, 0) > 3
            is_dynamic_header = report_header_pattern.match(txt)

            if not (is_freq_header or is_dynamic_header):
                 clean_blocks.append(b)
                 
        self.blocks = clean_blocks

    def process_content_structure(self):
        """Step 6: Content Processing (NLP Structuring & Metadata)"""
        print("Processing content structure...")
        
        doc_name = Path(self.pdf_path).stem
        
        # 🔥 HARDENING 5: Calculate Global Confidence for Metadata
        avg_conf = 1.0
        if self.blocks:
             avg_conf = round(sum(b.get('confidence', 1.0) for b in self.blocks) / len(self.blocks), 2)

        final_sections = []
        
        # 🔟 PROMPT FIX 7 & 10: Section Hierarchy & Semantic Purity
        # Track hierarchy: Theme (L1) -> Subtopic (L2) -> Case Study (L3)
        current_theme = "General"
        current_subtopic = "Introduction"
        current_case_study = None # If set, we are in a case study
        
        current_section = "Introduction"
        current_section_page = 1 
        current_buffer = [] 
        # 🔥 FIX: Page Tracking Set
        current_pages = set() 
        
        if self.blocks:
            current_section_page = self.blocks[0]['page']

        # Pre-process blocks for strict ordering and fixing
        processed_blocks = []
        for b in self.blocks:
             processed_blocks.append(b)
        
        i = 0
        while i < len(processed_blocks):
            block = processed_blocks[i]
            text = block['full_text'].strip()
            role = block.get('role', 'BODY')
            
            # 🔥 IMPROVEMENT 6: Enhanced Heading Detection
            if text.endswith(':'):
                role = 'HEADING'
                
            page = block['page']
            
            # --- 5️⃣ PROMPT FIX 5: Quote Isolation ---
            # "If line contains ", isolate: TYPE: EXECUTIVE_QUOTE SPEAKER: ..."
            # 🔥 FIX 3 & ISSUE 6: Strict Quote Detection (Requires attribution dash)
            if text.count('"') >= 2 and len(text) < 300 and re.search(r'–|—|-', text):
                # Flush current buffer
                 if current_buffer:
                    self._flush_section(final_sections, doc_name, current_theme, current_subtopic, current_case_study, current_pages, current_buffer)
                    current_buffer = []
                    current_pages = set()

                 # Extract Speaker (Heuristic: Look for "Says X" or "- Name")
                 speaker = "Unknown"
                 speaker_match = re.search(r'says\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)', text)
                 if speaker_match:
                     speaker = speaker_match.group(1)
                 
                 final_sections.append({
                    "id": f"{doc_name}_QUOTE_{i}",
                    "title": "Executive Quote",
                    "level": "QUOTE",
                    "page": page,
                    "content": text,
                    "speaker": speaker,
                    "type": "EXECUTIVE_QUOTE",
                    "importance": 1.5,
                    
                    # 🔥 FIX 2: ENTERPRISE METADATA FOR QUOTES
                    "source_pages": [page],
                    "source_page_start": page,
                    "source_page_end": page,
                    "page_span": str(page),
                    "section_order": len(final_sections) + 1,
                    "doc_name": doc_name,
                    "section_type": "QUOTE",
                    "semantic_role": "EXECUTIVE_QUOTE",
                    "retrieval_weight": 1.8,
                    # Fallback RAG Stats
                    "tokens_estimate": len(text.split()) * 1.3,
                    "source_file": Path(self.pdf_path).name,
                    "source_file": Path(self.pdf_path).name,
                    "doc_type": self.doc_type
                 })
                 i += 1
                 continue

            # --- 4️⃣ PROMPT FIX 4: Award/Fact Extraction ---
            # Heuristic: "Received X award", "Winner of", "Ranked #1"
            # Must contain specific keywords AND be short (< 50 words)
            text_lower = text.lower()
            award_keywords = ["award", "winner", "ranked", "recognized as", "best in class", "leader in"]
            has_keyword = any(k in text_lower for k in award_keywords)
            
            # 🔥 ISSUE 4: Stricter Award Source Check (Requires named org after by/from)
            has_source = bool(re.search(r'\b(by|from)\s+[A-Z]', text))
            
            # 🔥 FIX 5: Better Fact/Marketing Separator (Title Case Density)
            # Awards usually have Proper Nouns
            is_facty = len(text.split()) < 50 and sum(1 for w in text.split() if w.istitle()) > 2
    
            if has_keyword and has_source and is_facty and role == 'BODY':
                 self.award_blocks.append({
                     "text": text,
                     "page": page
                 })
                 i += 1
                 continue

            # --- 6️⃣ PROMPT FIX 6: Bullet Intelligence ---
            is_bullet = bool(self.bullet_pattern.match(text) or self.number_list_pattern.match(text) or text.startswith(("- ", "• ")))
            if is_bullet:
                 # 🔥 UPGRADE 3: Smarter Bullet Grouping
                 # If buffer has non-list content, flush it.
                 is_prev_bullet = False
                 if current_buffer:
                      # Check if last item looks like a bullet item (starts with bullet char)
                      if current_buffer[-1].strip().startswith("• "):
                          is_prev_bullet = True
                 
                 if current_buffer and not is_prev_bullet:
                     self._flush_section(final_sections, doc_name, current_theme, current_subtopic, current_case_study, current_pages, current_buffer)
                     current_buffer = []
                     current_pages = set()
                 
                 # Clean bullet char
                 clean_text = re.sub(r'^[\u2022\u2023\u25E6\u2043\u2219\u25AA\*\-]\s*', '', text)
                 
                 # Append with bullet char to maintain list structure in buffer & allow detection
                 current_buffer.append(f"• {clean_text}")
                 current_pages.add(page)
                 # Set type? We accept that it will be mixed into NARRATIVE unless we track type. 
                 # But grouping them keeps them together.
                 i += 1
                 continue
            
            # --- 3️⃣ PROMPT FIX 3: Case Study Splitter ---
            # Detect new company
            # Patterns: <Proper Noun> is a... | <Company>, headquartered... | <Company> is Switzerland's...
            cs_match = re.match(r'^([A-Z][a-zA-Z0-9\s&]+)(?:is a|is the|is one of|,\s*headquartered|is Switzerland)', text)
            if cs_match and len(text) > 40 and role == 'BODY':
                 company_name = cs_match.group(1).strip()
                 
                 # 🔥 FIX 3: Case study detector safety checks
                 # Must be multi-word and start with uppercase
                 is_valid_name = len(company_name.split()) >= 2 and company_name[0].isupper()
                 
                 # Reject generic words that match pattern but aren't companies
                 generic_starts = ("ai", "we", "this", "our", "the", "it", "digital", "technology")
                 is_generic = company_name.lower().startswith(generic_starts)
                 
                 # 🔥 UPGRADE 4: Case Study Safety (Common False Positives)
                 common_false = ('Digital', 'Technology', 'AI', 'Cloud', 'Strategy', 'Sustainability', 'Global', 'Artificial', 'Platform')
                 if company_name.split()[0] in common_false:
                     is_generic = True
                     
                 # 🔥 MICRO FIX 2: Additional Invalid Starts
                 invalid_cs_starts = ("Artificial", "Digital", "Technology", "AI", "Cloud", "Platform")
                 if company_name.split()[0] in invalid_cs_starts:
                     is_valid_name = False

                 # 🔥 IMPROVEMENT 7 & ISSUE 2: Robust Case Study Name Check
                 # Allow "PayPal" (Title case 1 word) or "Sun Microsystems" (2+ caps)
                 cap_words = sum(1 for w in company_name.split() if w[0].isupper())
                 is_valid_name = (cap_words >= 2) or (company_name.istitle() and len(company_name.split()) == 1)
                 
                 # Extra safety: Reject if known invalid
                 if company_name.split()[0] in common_false or company_name.split()[0] in invalid_cs_starts:
                     is_valid_name = False

                 if is_valid_name and not is_generic and len(company_name.split()) < 7:
                     # Flush
                     if current_buffer:
                        self._flush_section(final_sections, doc_name, current_theme, current_subtopic, current_case_study, current_pages, current_buffer)
                        current_buffer = []
                        current_pages = set()
                     
                     current_case_study = company_name
                     current_subtopic = f"Case Study: {company_name}"
                     # Add this text as start of case study
                     current_buffer.append(text)
                     current_pages.add(page)
                     i += 1
                     continue

            # --- 🔟 PROMPT FIX 10: Semantic Purity Rule (Check for Header) ---
            if role == 'HEADING':
                 # 🔥 FIX 2: Running Header Filter BEFORE Flush
                 if len(text.split()) < 3:
                     # likely running header, not section title
                     i += 1
                     continue

                 # 🔥 MICRO FIX 3: List -> Paragraph Merge Safety
                 # If buffer ends with list item, force flush before new heading
                 if current_buffer and current_buffer[-1].strip().startswith("• "):
                    self._flush_section(final_sections, doc_name, current_theme, current_subtopic, current_case_study, current_pages, current_buffer)
                    current_buffer = []
                    current_pages = set()

                 if current_buffer:
                    self._flush_section(final_sections, doc_name, current_theme, current_subtopic, current_case_study, current_pages, current_buffer)
                    current_buffer = []
                    current_pages = set()
                 
                 # Determine Hierachy
                 current_subtopic = text
                 current_case_study = None # Reset case study on new heading?
                 
                 # 🔥 FIX 4: Backup Case Study Logic (Multi-sentence Intros)
                 # If shorter than 5 words and looks like a name -> Potential Case Study Header
                 if len(text.split()) <= 5:
                     clean_head = re.sub(r'[^a-zA-Z\s]', '', text).strip()
                     is_valid_name = len(clean_head.split()) >= 2 and clean_head[0].isupper()
                     
                     # 🔥 HARDENING 3: Corporate Suffix Filter
                     # Reduce misfires on "Digital Transformation is a..."
                     corp_terms = ('Ltd', 'Inc', 'Corp', 'Group', 'Bank', 'Telecom', 'Systems', 'Technologies')
                     is_corp_like = any(t in clean_head for t in corp_terms)
                     
                     generic_starts = ("ai", "we", "this", "our", "the", "it", "digital", "technology", "strategy", "overview", "financial")
                     is_generic = clean_head.lower().startswith(generic_starts)
                     
                     # Require: Valid Name AND (Not Generic OR Has Corp Term)
                     if is_valid_name and (not is_generic or is_corp_like):
                         # Treat as Case Study
                         current_case_study = clean_head
                         current_subtopic = f"Case Study: {clean_head}"

                 # If Heading contains "Strategy" or "Theme", maybe it's Level 1?
                 if "Strategy" in text or "Theme" in text or "Chapter" in text:
                     current_theme = text
                 i += 1
                 continue

            # --- 2️⃣ PROMPT FIX 2: Paragraph Boundary Correction ---
            # Logic:
            # 1. Start with current text.
            # 2. Check overlap with next block for merge.
            
            # Add Tags (Fix 9)
            tags = []
            if '%' in text: tags.append('METRIC')
            if re.search(r'[\$₹€£]', text): tags.append('FINANCIAL')
            if re.search(r'\b(award|recognition|winner)\b', text.lower()): tags.append('RECOGNITION')
            if re.search(r'\b(employees|workforce)\b', text.lower()): tags.append('HR_METRIC')
            
            tagged_text = text
            if tags:
                 tagged_text = f"[{' | '.join(tags)}] {text}"
            
            # Merge Logic
            if not current_buffer:
                current_buffer.append(tagged_text)
                current_pages.add(page)
            else:
                # 🔥 UPGRADE 3: Reset Buffer if transitioning from List to Narrative
                if current_buffer[-1].strip().startswith("• "):
                    self._flush_section(final_sections, doc_name, current_theme, current_subtopic, current_case_study, current_pages, current_buffer)
                    current_buffer = []
                    current_pages = set()
                    current_buffer.append(tagged_text)
                    current_pages.add(page)
                    i += 1
                    continue

                last_text = current_buffer[-1]
                # Remove tags for analysis
                last_text_clean = re.sub(r'^\[.*?\]\s*', '', last_text)
                
                # 🔥 FIX 1: Define ends_sentence EARLY (Prevent Crash)
                ends_sentence = last_text_clean.strip().endswith(('.', '!', '?'))
                
                should_merge = False
                
                # Rule: Sentence starts lowercase -> Merge with previous
                if text[0].islower():
                    should_merge = True

                # 🔥 UPGRADE 2: Broken Sentence Merge (Lines ending without period)
                if not ends_sentence and text[0].isupper() and len(text.split()) < 12:
                    should_merge = True
                
                # 🔥 HARDENING 2: Transition Word Merge
                # Merge if line starts with transition word (However, But, etc.) even if capitalized
                transition_starts = ('However', 'But', 'And', 'So', 'Because', 'Therefore', 'Moreover', 'Thus')
                if text.startswith(transition_starts):
                    should_merge = True
                
                # Rule: Line starts with capital after full stop -> New Paragraph (Split)
                # (Implicit: If NOT should_merge, we split. So we just strictly check merge conditions)
                
                # Rule: Page number changes mid-sentence -> Merge
                # Checked by: text[0].islower() often handles "mid-sentence" flow.
                # Also check punctuation of previous block.
                # (ends_sentence check was here, now moved up)
                
                if not ends_sentence and not text[0].isupper():
                     should_merge = True

                # 🔥 UPGRADE 3: Hard Paragraph Split Rule
                new_para_starts = ('In addition', 'For example', 'As a result', 'On the other hand', 'However,')
                if text.startswith(new_para_starts):
                    should_merge = False
                
                # 🔥 FIX 3: Strict Cross-Column Merge Prevention & Vertical Gap
                # ⚠️ ISSUE 3: Use self.blocks to ensure we check against correct geometry
                # 🔥 FIX 3: Use processed_blocks for correct index alignment
                prev_block = processed_blocks[i-1] if i > 0 else None
                if prev_block:
                     # Column Mismatch
                     if block.get('column') != prev_block.get('column'):
                         should_merge = False
                     
                     # 🔥 FIX 1: Use Preserved Geometry (bbox is gone)
                     if 'y_top' in block and 'y_bottom' in prev_block:
                         vertical_gap = block['y_top'] - prev_block['y_bottom']
                         if vertical_gap > 15:
                             should_merge = False

                if should_merge:
                    # 🔥 FIX 2: Double merge bug fix - only merge once
                    current_buffer[-1] = last_text + " " + text
                    current_pages.add(page)
                else:
                    current_buffer.append(tagged_text)
                    current_pages.add(page)
            
            i += 1
            
        # Final Flush
        if current_buffer:
            self._flush_section(final_sections, doc_name, current_theme, current_subtopic, current_case_study, current_pages, current_buffer)
            current_buffer = []
            current_pages = set()
            
        self.output_text = final_sections

    def _flush_section(self, sections, doc_name, theme, subtopic, case_study, pages, buffer):
        if not buffer: return
        
        # Calculate Metadata EARLY for Filter
        level = "LEVEL 2"
        if case_study: level = "LEVEL 3 (CASE STUDY)"
        elif theme == subtopic: level = "LEVEL 1"
        
        imp = 1.0
        if "CASE STUDY" in level: imp = 1.2
        if "STRATEGY" in subtopic.upper(): imp = 1.5

        # 🔥 FIX 6: Semantic split enforcement
        # If buffer contains mixed content (strategy + case study intro), split it
        # This is a safety check - ideally the main loop prevents this
        if "Case Study:" in subtopic and len(buffer) > 3:
            pass # (Keep pass logic)
        
        content = "\n".join(buffer)
        
        # 🔥 UPGRADE 8: Sentence Repair (Quality Boost)
        content = re.sub(r'\s+([,.])', r'\1', content)   # space before punctuation
        content = re.sub(r'([a-z])([A-Z])', r'\1 \2', content)  # fix wordjoins
        
        # 🔥 IMPROVEMENT 5: Sentence Completion Repair (Newline to Period)
        content = re.sub(r'(?<![.!?])\n(?=[A-Z])', '. ', content)
        
        # 🔥 UPGRADE 2: Table-to-Text Boundary Repair
        content = re.sub(r'\.\s*\n\s*[a-z]', lambda m: m.group(0).replace('\n',' '), content)
        
        # 🔸 ISSUE 1: Safe Section Filter (Uses Importance)
        if len(content.split()) < 15 and imp < 1.2:
            return
        
        # 🔥 ENTERPRISE DATA: Uses Explicit Page Tracking
        if not pages:
            # Fallback to regex if pages set is empty (shouldn't happen)
            page_nums = [int(p) for p in re.findall(r'PG=(\d+)', content)]
        else:
             page_nums = sorted(list(pages))
             
        if not page_nums:
            page_nums = [1] # Safety
        
        start_page = min(page_nums)
        end_page = max(page_nums)
        page_span = f"{start_page}-{end_page}" if start_page != end_page else str(start_page)

        # ID Generation
        clean_sub = re.sub(r'[^a-zA-Z0-9]+', '_', subtopic).upper()[:30]
        stable_id = f"{doc_name}_SEC_{clean_sub}_{len(sections)}"
        
        # (Hierarchy & Imp was computed above)
        
        # 🔥 ENTERPRISE METADATA INJECTION
        sections.append({
            "id": stable_id,
            "title": subtopic,
            "theme": theme,
            "case_study": case_study,
            "level": level,
            "page": start_page, # Legacy support
            "content": content,
            "importance": imp,
            "type": "NARRATIVE",
            
            # Enterprise Traceability
            "source_pages": page_nums, # List of all pages involved
            "source_page_start": start_page,
            "source_page_end": end_page,
            "page_span": page_span,
            "section_order": len(sections) + 1,
            "doc_name": doc_name,
            "section_type": level,
            "semantic_role": "CASE_STUDY" if case_study else "NARRATIVE",
            "retrieval_weight": round(imp * 1.2, 2),
            
            # RAG Stats
            "tokens_estimate": int(len(content.split()) * 1.3),
            "source_file": Path(self.pdf_path).name,
            "doc_type": self.doc_type,
            "confidence_avg": round(sum(b.get('confidence', 1.0) for b in self.blocks) / len(self.blocks), 2) if self.blocks else 1.0
        })

        # 🔥 MICRO FIX 1 & ISSUE 5: Correct Weighted Section Confidence Logic
        relevant_blocks = [b for b in self.blocks if b['page'] in page_nums]
        if relevant_blocks:
            # Weighted by text length
            total_len = sum(len(b.get('full_text','').split()) for b in relevant_blocks)
            if total_len > 0:
                weighted_sum = sum(b.get('confidence', 1.0) * len(b.get('full_text','').split()) for b in relevant_blocks)
                section_conf = round(weighted_sum / total_len, 2)
            else:
                section_conf = 1.0
        else:
            section_conf = 1.0
            
        # Update just added dict
        sections[-1]["confidence_section"] = section_conf

    def integrate_awards_into_narrative(self):
        """Step 6.5: Transform Awards to Narrative and Integrate"""
        if not self.award_blocks:
            return

        print("Transforming awards into narrative structure...")
        
        narrative_lines = []
        for award in self.award_blocks:
            page = award["page"]
            text = award["text"]
            
            # 1. Extract year
            year_match = re.search(r'(20\d{2})', text)
            year = year_match.group(1) if year_match else "the reported year"
            
            # 2. Clean text
            clean_text = re.sub(r'^[\u2022\u2023\u25E6\u2043\u2219\*\-\s]+', '', text).strip()
            
            # Avoid double naming if text starts with Infosys
            if clean_text.lower().startswith("infosys"):
                clean_text = clean_text[7:].strip()
            
            # 3. Format: [FACT | AWARD | PG={page}]
            semantic_sentence = f"In {year}, Infosys {clean_text}. [FACT | AWARD | PG={page}]"
            narrative_lines.append(semantic_sentence)
            
        if narrative_lines:
            # Add as a new section
            section_title = "Recognitions & Industry Standing"
            start_page = self.award_blocks[0]["page"]
            content = "\n".join(narrative_lines)
            
            clean_title = "RECOGNITIONS_INDUSTRY_STANDING"
            stable_id = f"{Path(self.pdf_path).stem}_SEC_{clean_title}"
            
            # 🔥 FIX 3: AWARD PAGE SPAN TRACKING
            award_pages = sorted(set(a["page"] for a in self.award_blocks))
            if not award_pages:
                 award_pages = [start_page]
                 
            start_page = award_pages[0]
            end_page = award_pages[-1]
            page_span = f"{start_page}-{end_page}" if start_page != end_page else str(start_page)

            self.output_text.append({
                "id": stable_id,
                "title": section_title,
                "page": start_page,
                "content": content,
                "importance": 0.8, # Facts are good but maybe less critical than strategy
                "type": "FACT_LIST",
                
                # 🔥 ENTERPRISE METADATA FOR AWARDS
                "source_pages": award_pages,
                "source_page_start": start_page,
                "source_page_end": end_page,
                "page_span": page_span,
                "section_order": len(self.output_text) + 1,
                "doc_name": Path(self.pdf_path).stem,
                "section_type": "FACT_LIST",
                "semantic_role": "FACTS",
                "retrieval_weight": 0.9,
                # Fallback RAG Stats
                "tokens_estimate": int(len(content.split()) * 1.3),
                "source_file": Path(self.pdf_path).name,
                "doc_type": getattr(sys.modules[__name__], 'DOC_TYPE', 'GENERAL')
            })

    def generate_narrative_output(self, output_file):
        """Step 7: Output Generation"""
        print(f"Generating output to {output_file}...")
        
        if not self.output_text:
            print("Warning: No content to write.")
            return

        with open(output_file, "w", encoding="utf-8") as f:
            for sec in self.output_text:
                level = sec.get('level', 'LEVEL 1') 
                # 🔥 FIX 1: SHOW PAGE SPAN IN TXT HEADER
                header = f"[SECTION ID: {sec['id']} | LEVEL: {level} | IMPORTANCE: {sec['importance']:.1f}] {sec['title']} (Pages: {sec.get('page_span', sec['page'])})"
                f.write(header + "\n")
                f.write(sec['content'] + "\n\n")
                f.write("-" * 40 + "\n\n")
        
        
        print("Done.")

    # 🔥 FIX 6: Add JSON Export for RAG
    def export_json(self, output_file):
        """Step 8: JSON Export for RAG Ingestion"""
        print(f"Exporting structured JSON to {output_file}...")
        
        if not self.output_text:
             print("Warning: No content to export.")
             return

        with open(output_file, 'w', encoding='utf-8') as f:
             json.dump(self.output_text, f, indent=2, ensure_ascii=False)
        
        print("Done.")

    def print_extraction_stats(self):
        """Step 9: Observability & Stats"""
        print("\n" + "="*40)
        print("   🔍 DOCUMENT INTELLIGENCE REPORT   ")
        print("="*40)
        print(f" Pages Processed         : {self.stats['pages_processed']}")
        print(f" Narrative Pages         : {self.stats['narrative_pages']}")
        print(f" Table Pages Skipped     : {self.stats['table_pages_skipped']}")
        print(f" Total Blocks Analyzed   : {self.stats['blocks_processed']}")
        print(f" Duplicates Removed      : {self.stats['duplicate_blocks_removed']}")
        print(f" Low Confidence Blocks   : {self.stats['low_confidence_blocks']}")
        print(f" Extracted Sections      : {len(self.output_text)}")
        print(f" Extracted Facts (Awards): {len(self.award_blocks)}")
        print("="*40 + "\n")

def run_extraction(pdf_path, skip_pages=None, output_folder=None):
    """Run pipeline for a specific configuration"""
    # --------------------------------------------------
    # ARGUMENT STANDARDIZATION
    # --------------------------------------------------
    if skip_pages is None:
        skip_pages = set()
    
    if output_folder is None:
        # Default output if not provided
        output_folder = r"C:\financial_agent\outputs\financial_extraction"

    input_path = pdf_path
    table_pages = skip_pages
    output_subfolder = output_folder
    
    print(f"\n{'='*50}")
    print(f"STARTING JOB: {str(Path(input_path).name)}")
    print(f"Output Folder: {output_subfolder}")
    print(f"{'='*50}")
    
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    # Setup Output
    input_filename = Path(input_path).stem
    output_dir = output_subfolder
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{input_filename}.txt")
    json_output_file = os.path.join(output_dir, f"{input_filename}.json")

    # Execution Pipeline
    extractor = PDFNarrativeExtractor(input_path, table_pages_to_skip=table_pages)
    try:
        extractor.load_pdf()
        extractor.detect_reading_order()    
        extractor.detect_section_boundaries() 
        extractor.filter_text_blocks()
        extractor.remove_headers_footers()
        extractor.process_content_structure() 
        extractor.integrate_awards_into_narrative()
        extractor.generate_narrative_output(output_file)
        extractor.export_json(json_output_file)
        extractor.print_extraction_stats()
        print(f"\nSuccess! Output saved to:\n{output_file}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    return output_file

if __name__ == "__main__":
    # Execute only the selected PDF
    print(f"🎯 Target PDF: {Path(SELECTED_PDF).name}")
    
    # Find config matching the selection
    target_config = next((c for c in CONFIGS if os.path.normpath(c["path"]) == os.path.normpath(SELECTED_PDF)), None)
    
    if target_config:
        print("✅ Found matching configuration.")
        run_extraction(
            pdf_path=target_config["path"],
            skip_pages=target_config.get("skip_pages"),
            output_folder=target_config["output_folder"]
        )
    else:
        # Fallback for new/unconfigured files
        print("⚠️ No specific config found - Running generic extraction to 'output/generic'")
        run_extraction(
             pdf_path=SELECTED_PDF,
             skip_pages=set(),
             output_folder=r"C:\financial_agent\outputs\financial_extraction"
        )
