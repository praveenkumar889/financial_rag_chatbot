import os
import json
import base64
import sys
from pathlib import Path
import fitz  # pymupdf
from collections import Counter
import re
from dotenv import load_dotenv, find_dotenv
from openai import AzureOpenAI

# Load environment variables from the root .env
load_dotenv(find_dotenv(), override=True)

# Azure Configuration
AZURE_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_AI_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

if not AZURE_ENDPOINT or not AZURE_API_KEY:
    exit(1)

client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version="2024-02-15-preview"
)

# Configuration
CONFIG = {
    "default_pdf_path": r"C:\financial_agent\input_pdf\infy-ppt_Q3_FY26.pdf",
    "output_dir": r"C:\financial_agent\outputs\ppt_extraction",
    "zoom_level": 2
}

def encode_image(pix):
    """Encodes a PyMuPDF pixmap to base64."""
    img_bytes = pix.tobytes("png")
    return base64.b64encode(img_bytes).decode('utf-8')

def extract_metadata_from_filename(file_path):
    """
    Extract Fiscal Year and Quarter from full file path.
    Supports formats like:
    - Q3_FY26
    - FY26_Q3
    - Q3FY26
    - FY2026-Q3
    """
    filename = Path(file_path).stem.upper()

    fy_match = re.search(r'FY\s?(\d{2,4})', filename)
    q_match = re.search(r'Q\s?([1-4])', filename)

    fiscal_year = f"FY{fy_match.group(1)}" if fy_match else None
    quarter = f"Q{q_match.group(1)}" if q_match else None

    return fiscal_year, quarter



def enrich_table_metadata(page_data, document_metadata):
    fy = document_metadata.get("fiscal_year", "Unknown")
    q = document_metadata.get("quarter", "Unknown")

    time_period = []
    if q != "Unknown" and fy != "Unknown":
        time_period.append(f"{q}_{fy}")
    if fy != "Unknown":
        time_period.append(fy)
    
    for table in page_data.get("tables", []):
        title = table.get("title", "").lower()

        if "revenue" in title:
            category = "revenue"
            statement = "income_statement"
        elif "cash flow" in title:
            category = "cash_flow"
            statement = "cash_flow"
        elif "profit" in title or "income" in title:
            category = "profitability"
            statement = "income_statement"
        else:
            category = "other"
            statement = "unknown"

        table["semantic_metadata"] = {
            "content_type": "table_analysis",
            "financial_statement": statement,
            "metric_category": category,
            "time_period": time_period,
            "granularity": "page_level"
        }

def enrich_chart_metadata(page_data, document_metadata):
    fy = document_metadata.get("fiscal_year", "Unknown")
    q = document_metadata.get("quarter", "Unknown")

    time_period = []
    if q != "Unknown" and fy != "Unknown":
        time_period.append(f"{q}_{fy}")

    for chart in page_data.get("charts", []):
        chart["semantic_metadata"] = {
            "content_type": "chart_analysis",
            "time_period": time_period,
            "granularity": "page_level"
        }

def add_financial_analysis(page_data):
    """
    Derives financial insights from extracted key metrics.
    """
    insights = []
    # Join key metrics into a single string for keyword searching
    metrics = " ".join(page_data.get("key_metrics", []))

    # Example: Cash Conversion
    if "Free Cash Flow" in metrics and "Operating Profit" in metrics:
        insights.append("Free cash flow closely tracks operating profit, indicating strong cash conversion efficiency.")

    # Example: Margin
    if "Operating Margin" in metrics:
        insights.append("Operating margin above 20% indicates strong profitability for an IT services company.")

    # Example: ROE
    if "ROE" in metrics:
        insights.append("High ROE suggests efficient capital utilization and strong shareholder returns.")
    
    # Example: Growth
    if "Growth" in metrics or "YoY" in metrics:
         insights.append("Year-over-year growth metrics indicate business expansion trends.")

    if insights:
        page_data["derived_financial_insight"] = " ".join(insights)
    else:
        page_data["derived_financial_insight"] = "No derived financial insight."

def extract_page_data(page_num, base64_image):
    """Sends page image to Azure OpenAI for structured extraction."""
    
    system_prompt = """
    You are an advanced Financial Document Analysis AI. 
    Your task is to analyze the provided page from a financial presentation and extract data into a strict JSON structure.
    
    The desired structure matches this hierarchy:
    1. Layout Analysis: Detect headings, sections, tables, charts.
    2. Text Blocks: Extract text in Markdown format, preserving hierarchy.
    3. Tables: Parse structured data, context metadata, AND generate a detailed narrative analysis.
    4. Charts (Bar/Line/Pie): Extract data points and generate a narrative.
    5. Infographics/Diagrams: Provide a structural summary.

    IMPORTANT:
    5. Infographics/Diagrams: Provide a structural summary.

    IMPORTANT:
    You must:
    - Extract all visible financial values
    - Calculate simple relationships when possible
    - Explain trends (growth, decline, stability)
    - Identify strengths (margin, ROE, FCF, growth)
    - Identify risk signals (decline, concentration, volatility)
    - Keep insights factual and based only on visible data
    - Avoid generic statements

    Output MUST be a valid JSON object with the following schema:
    {
      "page_number": integer,
      "page_summary": "Brief summary of the page content",
      "strategic_insight": "Brief analytical interpretation of the page. Explain financial strength, risks, growth drivers, or implications based on extracted data. Avoid generic statements.",
      "key_metrics": [
        "String containing specific metric and value (e.g., 'Revenue: $19.8 bn', 'ROE: 32.8%')"
      ],
      "layout_analysis": {
        "headings": [strings],
        "sections": [strings],
        "detected_elements": ["table", "chart", "text", "infographic", etc]
      },
      "text_blocks": [
        { "content": "markdown text", "type": "paragraph/heading/list" }
      ],
      "tables": [
        {
          "title": "string",
          "data": [[row1_col1, row1_col2], ...],
          "metadata": "context/units",
          "narrative": "Compact narrative including specific values (e.g., 'Revenue grew 10% to $5M')."
        }
      ],
      "charts": [
        {
          "type": "bar/line/pie/etc",
          "title": "string",
          "data_points": [{"label": "x", "value": "y"}, ...],
          "narrative": "Compact narrative including specific values and trends (e.g., 'Market share rose to 25%')."
        }
      ],
      "infographics": [
        {
          "description": "string",
          "structural_summary": "Detailed explanation of the visual flow and meaning"
        }
      ]
    }
    
    You must also generate a field called "strategic_insight".
    This should interpret the numbers and trends.
    Do not repeat metrics.
    Explain implications, strengths, risks, or business meaning.
    If no financial data exists, say "No strategic financial insight on this page."
    Do not include any markdown formatting (like ```json) in the response, just the raw JSON string.
    """

    user_prompt = f"Analyze this page (Page {page_num}) and extract the data."

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0.0,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error extracting page {page_num}: {e}")
        return None

def save_narrative_text(data, output_file, metadata):
    """Generates a text file with narrative analysis from the extracted data."""
    narrative_lines = []
    
    company = metadata.get("company", "Unknown")
    fy = metadata.get("fiscal_year", "Unknown")
    quarter = metadata.get("quarter", "Unknown")
    
    for page in data:
        page_num = page.get("page_number", "?")
        
        # Add Metadata Header for RAG ingestion
        # Format: [METADATA | Key=Val | Key2=Val2]
        # This one is tricky because we construct a list of strings, so we add it to list
        narrative_lines.append(f"[METADATA | Page={page_num} | FY={fy} | Quarter={quarter} | Company={company}]")
        
        narrative_lines.append(f"--- Analysis of Page {page_num} ---")
        
        # Page Summary
        if page.get("page_summary"):
             narrative_lines.append(f"Summary: {page.get('page_summary')}\n")

        # Key Metrics
        key_metrics = page.get("key_metrics", [])
        if key_metrics:
            narrative_lines.append("### Key Metrics & Highlights:")
            for metric in key_metrics:
                narrative_lines.append(f"- {metric}")
            narrative_lines.append("")

        # Tables
        tables = page.get("tables", [])
        if tables:
            for i, table in enumerate(tables):
                title = table.get("title", f"Table {i+1}")
                narrative = table.get("narrative", "No narrative available.")
                narrative_lines.append(f"### Table Analysis: {title}")
                narrative_lines.append(f"{narrative}\n")

        # Charts
        charts = page.get("charts", [])
        if charts:
            for i, chart in enumerate(charts):
                title = chart.get("title", f"Chart {i+1}")
                narrative = chart.get("narrative", "No narrative available.")
                narrative_lines.append(f"### Chart Analysis: {title}")
                narrative_lines.append(f"{narrative}\n")

        # Infographics
        infographics = page.get("infographics", [])
        if infographics:
            for i, info in enumerate(infographics):
                desc = info.get("description", "Infographic")
                summary = info.get("structural_summary", "")
                narrative_lines.append(f"### Infographic: {desc}")
                narrative_lines.append(f"{summary}\n")

        # Strategic Insight
        if page.get("strategic_insight"):
            narrative_lines.append("### Strategic Insight:")
            narrative_lines.append(page.get("strategic_insight"))
            narrative_lines.append("")

        # Derived Insight
        if page.get("derived_financial_insight"):
            narrative_lines.append("### Derived Financial Insight:")
            narrative_lines.append(page.get("derived_financial_insight"))
            narrative_lines.append("")

        narrative_lines.append("=" * 50 + "\n")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(narrative_lines))
        print(f"Narrative text saved to {output_file}")
    except Exception as e:
        print(f"Error saving narrative text: {e}")

def run_ppt_extraction(pdf_path, output_dir=None):
    if output_dir is None:
        output_dir = CONFIG["output_dir"]

    # Clean up the path (remove quotes if pasted)
    pdf_path = pdf_path.strip('"').strip("'")

    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at: {pdf_path}")
        return None

    # Derive output filenames from the PDF name
    pdf_name = Path(pdf_path).stem
    os.makedirs(output_dir, exist_ok=True)
    
    json_output_path = os.path.join(output_dir, f"{pdf_name}_extracted_data.json")
    narrative_output_path = os.path.join(output_dir, f"{pdf_name}_narrative.txt")

    print(f"Processing PDF: {pdf_path}")
    print(f"Output JSON: {json_output_path}")
    print(f"Output Narrative: {narrative_output_path}")

    doc = fitz.open(pdf_path)
    
    all_extracted_data = []
    
    # 🔥 Step: Auto Detect Primary FY + Quarter from Filename - AUTHORITATIVE
    filename_fy, filename_q = extract_metadata_from_filename(pdf_path)

    if not filename_fy or not filename_q:
        print(f"Error: Filename '{pdf_path}' must contain FY and Quarter (e.g., Q3_FY26).")
        # You could raise an error or exit here. For now, we will exit to enforce cleanliness.
        return

    print(f"Filename Metadata Authoritative: {filename_fy}, {filename_q}")

    document_metadata = {
        "company": re.split(r'[-_]', pdf_name)[0].upper(),  # Robust company name extraction
        "document_name": pdf_name,
        "document_type": "presentation",
        "fiscal_year": filename_fy,
        "quarter": filename_q,
        "currency": ["USD"],
        "source": "presentation",
        "language": "en"
    }

    # Process each page
    for i in range(len(doc)):
        page = doc[i]
        print(f"Processing Page {i + 1}/{len(doc)}...")
        
        # Render page to image (zoom=2 for better quality)
        pix = page.get_pixmap(matrix=fitz.Matrix(CONFIG["zoom_level"], CONFIG["zoom_level"]))
        img_base64 = encode_image(pix)
        


        page_data = extract_page_data(i + 1, img_base64)
        if page_data:
            # Collect detected periods for global consensus (Removed: No longer needed)
             
            # Inject Metadata Placeholders (will be finalized in post-processing)
            page_data["page_metadata"] = {
                "page_number": page_data.get("page_number"),
                "content_types_present": page_data.get("layout_analysis", {}).get("detected_elements", []),
                "has_tables": bool(page_data.get("tables")),
                "has_charts": bool(page_data.get("charts")),
                "has_infographics": bool(page_data.get("infographics"))
            }
            
            # Note: Enrichment and Metadata Injection moved to final pass for efficiency
            all_extracted_data.append(page_data)

    doc.close()


    print(f"Using Authoritative Filename Metadata: {filename_fy}, {filename_q}")

    # 🔥 Re-Inject Finalized Metadata & Run Enrichment Logic
    # This ensures that even pages extracted BEFORE the global consensus was reached
    # (or if consensus changed things) now have the correct global context.
    # We also run financial analysis here to keep "Extraction" and "Enrichment" separate.
    print("Re-injecting finalized metadata and executing enrichment...")
    for page_data in all_extracted_data:
        # 1. Update Global Metadata
        page_data["document_metadata"] = document_metadata
        
        # 2. Update Table & Chart Metadata (Dependent on Global Metadata)
        enrich_table_metadata(page_data, document_metadata)
        enrich_chart_metadata(page_data, document_metadata)
        
        # 3. Derive Financial Insights (Dependent on Page Content)
        add_financial_analysis(page_data)

    # Save initial extraction
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_extracted_data, f, indent=2)
    print(f"Extraction complete. Data saved to {json_output_path}")

    # Generate Narrative Text File
    print("Generating narrative text file...")
    save_narrative_text(all_extracted_data, narrative_output_path, document_metadata)
    
    return narrative_output_path

def main():
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = CONFIG["default_pdf_path"]
        
    run_ppt_extraction(pdf_path, CONFIG["output_dir"])

if __name__ == "__main__":
    main()
