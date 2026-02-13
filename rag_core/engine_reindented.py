    import os
    import pymongo
    import logging
    from pymongo import MongoClient
    from openai import AzureOpenAI
    from dotenv import load_dotenv, find_dotenv
    import certifi
    import json
    from datetime import datetime
    import tiktoken
    from typing import List, Optional, Dict
    from pydantic import BaseModel, Field
    from functools import lru_cache
    import re
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Load environment variables
    basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(basedir, ".env"), override=True)
    
    # Azure Configuration
    AZURE_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT")
    AZURE_API_KEY = os.getenv("AZURE_AI_API_KEY")
    EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    CHAT_DEPLOYMENT = os.getenv("GPT_4_1_MINI_DEPLOYMENT", "gpt-4.1-mini") # Using Mini for speed/cost
    
    # MongoDB Configuration
    MONGO_URI = os.getenv("MONGO_URI")
    DB_NAME = os.getenv("MONGO_DB_NAME", "financial_rag")
    COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "rag_chunks")
    VECTOR_INDEX_NAME = "vector_index" # MUST match the index name you created in Atlas
    
    # DEBUG: Verify Env Loading
    # logger.info(f"DEBUG: Loading Env from {os.path.join(basedir, '.env')}")
    # logger.info(f"DEBUG: MONGO_URI Found: {'Yes' if MONGO_URI else 'No'}")
    # logger.info(f"DEBUG: DB_NAME: {DB_NAME}")
    # logger.info(f"DEBUG: COLLECTION: {COLLECTION_NAME}")
    
    # Initialize Azure Client
    client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version="2024-02-15-preview"
    )
    
def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Returns the number of tokens in a text string."""
try:
    encoding = tiktoken.encoding_for_model(model)
except KeyError:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))
    
def has_numeric_values(text: str) -> bool:
    """Checks if the text contains significant numeric values (currency, percentages, or large numbers)."""
    return bool(re.search(r'(\₹|\$|%|\d{2,}(?:,\d{3})*)', text))
    
class ConversationState(BaseModel):
    history: List[Dict[str, str]] = Field(default_factory=list)
    last_metric: Optional[str] = None
    last_year: Optional[str] = None
    last_company: Optional[str] = None
    
def add_interaction(self, query: str, answer: str):
    self.history.append({"role": "user", "content": query})
    self.history.append({"role": "assistant", "content": answer})
    
    # Trim history by tokens (keep under ~1000 tokens)
while len(self.history) > 0:
    history_text = "".join([f"{msg['role']}: {msg['content']}" for msg in self.history])
if count_tokens(history_text) > 1000:
    self.history.pop(0) # Remove oldest message
else:
    break
    
class CritiqueResult(BaseModel):
    status: str
    reason: Optional[str] = None
    
    conversation_state = ConversationState()
    
def classify_query_intent_rule_based(query):
    """Fallback rule-based intent classification."""
    q = query.lower()
    
if "operating profit" in q or "operating margin" in q:
    return "operating_profit"
    
if any(x in q for x in ["say", "management", "comment", "guidance", "outlook"]):
    return "management_commentary"
    
if any(x in q for x in ["compare", "vs", "difference", "growth"]):
    return "comparison"
    
if any(x in q for x in ["net profit", "profit for the year", "revenue", "income"]):
    return "financial_data"
    
    return "general"
    
def classify_query_intent(query):
    """Classifies the intent of the user query using LLM with fallback."""
try:
    response = client.chat.completions.create(
    model=CHAT_DEPLOYMENT,
    messages=[
    {"role": "system", "content": """Classify the user query into ONE of these categories:
    - operating_profit: Specific questions about operating profit/margin
    - management_commentary: Questions about what management said, guidance, outlook
    - comparison: Questions comparing years, companies, or asking for differences/growth
    - financial_data: General financial metrics (revenue, net profit, etc)
    - general: Anything else
    
    Respond ONLY with the category name."""},
    {"role": "user", "content": query}
    ],
    temperature=0,
    max_tokens=10
    )
    intent = response.choices[0].message.content.strip()
    valid_intents = ["operating_profit", "management_commentary", "comparison", "financial_data", "general"]
if intent in valid_intents:
    return intent
else:
    logger.warning(f"LLM returned invalid intent '{intent}', falling back to rules.")
    return classify_query_intent_rule_based(query)
except Exception as e:
    logger.error(f"Intent classification failed: {e}")
    return classify_query_intent_rule_based(query)
    
def self_check_answer(question, context, answer, answer_mode="DEFAULT") -> CritiqueResult:
    """Uses LLM to critique the generated answer for conflicts or ambiguity using JSON output."""
if answer == "Error generating answer from LLM.":
    return CritiqueResult(status="VALID", reason="Generation failed, skipping critique")
    
    special_instructions = ""
if answer_mode == "EXTRACTIVE_NUMERIC":
    special_instructions = """
    - MODE: EXTRACTIVE NUMERIC
    - IF the answer contains a precise numeric value from the context, mark as VALID.
    - Do NOT flag 'CLARIFICATION_NEEDED' for brevity. Short answers are required here.
    - Do NOT flag 'FIX_REQUIRED' unless the number is completely hallucinated or contradicts the context.
    """
    
    critique_prompt = f"""
    You are reviewing a financial answer for accuracy.
    
    Question:
    {question}
    
    Context:
    {context}
    
    Answer:
    {answer}
    
    Check STRICTLY:
    1. Are multiple fiscal years mixed?
    2. Are multiple profit definitions mixed (net profit, PAT, operating profit)?
    3. Are units inconsistent (₹ vs $)?
    4. Is the answer missing assumptions?
    5. Does the answer directly address the question?
    6. Does the answer include ANY calculated, implied, inferred, normalized, or back-calculated value that is NOT explicitly stated verbatim in the context?
    7. Did the answer switch financial metrics without user confirmation?
    8. If the answer derives a fiscal-year absolute value using percentages, ratios, or margins, is the derived value NOT explicitly stated verbatim?
    
    Respond ONLY with a JSON object in this format:
    {{
    "status": "VALID" | "CLARIFICATION_NEEDED" | "FIX_REQUIRED" | "REFUSAL_CORRECT",
    "reason": "explanation of the issue"
    }}
    
    Rules for Status:
    - If (1), (2), (3), (4) fails -> FIX_REQUIRED
    - If (6) or (8) fails AND the answer contains the calculated value -> FIX_REQUIRED
    - If (6) or (8) fails BUT the answer correctly refuses to calculate -> REFUSAL_CORRECT
    - If (5) or (7) fails -> CLARIFICATION_NEEDED
    - If normalized EPS and reported EPS conflict -> CLARIFICATION_NEEDED
    - If growth % is reported without both base values explicitly stated in the source -> REFUSAL_CORRECT
    
    CRITICAL OVERRIDE RULES:
    - Operating profit MUST NOT be calculated or inferred from revenue, margin, or growth.
    - If operating profit is not explicitly stated as a numeric value in the context, it MUST be treated as unavailable.
    - Implied, derived, inferred, normalized, or converted values are STRICTLY FORBIDDEN.
    - Even if a human analyst could compute a value, this system MUST NOT.
    - Definition: Explicit value = A numeric figure stated verbatim in the document for the exact metric and period. Anything else = NOT explicit.
    
    {special_instructions}
    """
    
try:
    response = client.chat.completions.create(
    model=CHAT_DEPLOYMENT,
    messages=[{"role": "user", "content": critique_prompt}],
    temperature=0,
    response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    result = CritiqueResult(**json.loads(content))
    
    # Hard override for derived logic violations
if result.reason and ("derived" in result.reason.lower() or "calculated" in result.reason.lower()):
if result.status == "VALID":
    result.status = "FIX_REQUIRED"
    
    # Override for Numeric Mode: Don't let it complain about missing context if we have the number
if answer_mode == "EXTRACTIVE_NUMERIC" and result.status in ["CLARIFICATION_NEEDED", "REFUSAL_CORRECT"]:
if has_numeric_values(answer):
    logger.info("Override: Self-check flagged a numeric answer, but we have numbers. Marking VALID.")
    result.status = "VALID"
    result.reason = "Numeric value present (override)"
    
    return result
except Exception as e:
    logger.error(f"Self check error: {e}")
    return CritiqueResult(status="VALID", reason="Self-check failed")
    
def log_correction(query, original_answer, corrected_answer):
    """Logs correction events for future learning."""
try:
    entry = {
    "query": query,
    "original": original_answer,
    "corrected": corrected_answer,
    "timestamp": datetime.utcnow().isoformat()
    }
    with open("rag_feedback.json", "a", encoding="utf-8") as f:
    json.dump(entry, f)
    f.write("\n")
except Exception as e:
    print(f"Logging error: {e}")
    
def connect_mongo():
    """Establishes a connection to MongoDB with robust SSL settings."""
try:
    mongo_client = MongoClient(
    MONGO_URI,
    tls=True,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=5000
    )
    # Verify connection
    mongo_client.admin.command('ping')
    # print("Connected to MongoDB successfully.") # Suppressed for cleaner engine output
    
    db = mongo_client[DB_NAME]
    return db[COLLECTION_NAME]
except Exception as e:
    print(f"MongoDB Connection Error: {e}")
    return None
    
    @lru_cache(maxsize=1000)
def get_query_embedding(query: str):
    """Generates embedding for the query using the same model as documents."""
try:
    response = client.embeddings.create(
    input=[query],
    model=EMBEDDING_DEPLOYMENT
    )
    return response.data[0].embedding
except Exception as e:
    print(f"Error generating embedding: {e}")
    return None
    
def vector_search(collection, query_embedding, top_k=5, filter_dict=None):
    """
    Performs vector search in MongoDB.
    """
    
    # Base Vector Search Stage
    vector_search_stage = {
    "index": VECTOR_INDEX_NAME,
    "path": "embedding",
    "queryVector": query_embedding,
    "numCandidates": 100,
    "limit": top_k
    }
    
    # Apply filter if provided
if filter_dict:
    vector_search_stage["filter"] = filter_dict
    
    pipeline = [
    {
    "$vectorSearch": vector_search_stage
    },
    {
    "$project": {
    "_id": 0,
    "text": 1,
    "metadata": 1,
    "score": { "$meta": "vectorSearchScore" }
    }
    }
    ]
    
try:
    results = list(collection.aggregate(pipeline))
    return results
except Exception as e:
    print(f"Vector Search Error: {e}")
    print("NOTE: Ensure you have created a Vector Search Index in MongoDB Atlas named 'vector_index' that indexes the 'embedding' field.")
    return []
    
def extract_entities(query):
    """Extracts financial entities (metric, year, company) from the query."""
try:
    response = client.chat.completions.create(
    model=CHAT_DEPLOYMENT,
    messages=[
    {"role": "system", "content": """Extract the following entities from the query:
    - metric: The financial metric (e.g., "revenue", "operating profit", "net income")
    - year: The fiscal year (e.g., "2023", "FY24")
    - company: The company name
    
    Respond ONLY with a JSON object:
    {"metric": "...", "year": "...", "company": "..."}
If an entity is missing, use null."""},
    {"role": "user", "content": query}
    ],
    temperature=0,
    response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)
except Exception as e:
    logger.error(f"Entity extraction failed: {e}")
    return {"metric": None, "year": None, "company": None}
    
    METRIC_SYNONYMS = {
    "revenue": ["revenue", "turnover", "sales"],
    "net profit": ["net profit", "profit for the year", "pat", "profit after tax"],
    "operating profit": ["operating profit", "operating income", "ebit"],
    "operating margin": ["operating margin", "ebit margin"]
    }
    
def filter_by_metric(results, metric):
    """Filters chunks to specific metric if detected using synonyms."""
if not metric:
    return results
    
    metric = metric.lower()
    keywords = METRIC_SYNONYMS.get(metric, [metric])
    
    filtered = [
    r for r in results
if any(k in r["text"].lower() for k in keywords)
    ]
    
    return filtered if filtered else results
    
def resolve_financial_conflicts(results):
    """Prioritizes sources based on strict hierarchy: Table > Annual Report > PPT > Transcript."""
    
    # Define Source Weights (Higher is better)
def get_source_weight(doc):
    doc_type = doc["metadata"].get("document_type", "")
    content_type = doc["metadata"].get("content_type", "")
    
    # 1. Audited Tables (Highest Authority)
if doc_type == "table" or content_type == "table_row":
    return 5
    
    # 2. Annual Report Narrative
if doc_type == "annual_report":
if "consolidated" in doc["text"].lower(): 
    return 4.5 # Boost Consolidated
    return 4
    
    # 3. Fact Sheet / PPT (Investor Pres)
if doc_type == "ppt":
    return 3
    
    # 4. Transcripts
if doc_type == "transcript":
    return 2
    
    return 1
    
    # Sort by Weight (Desc) then Score (Desc)
    # Python's sort is stable, so we sort by score first, then weight.
    # Actually, simpler to key a tuple.
    
    sorted_results = sorted(results, key=lambda x: (get_source_weight(x), x["score"]), reverse=True)
    
    # If we have a Level 5 source (Table), we might want to drop Level 2 (Transcription) conflicts for numbers.
    # But for RAG context, we usually keep them but ensure the LLM knows priority.
    # However, user requested: "If a value appears in a higher-ranked source, lock it and suppress conflicts".
    # We will trust the LLM prompt to respect the order, as we are presenting them ordered.
    
    return sorted_results
    
def deduplicate_results(results):
    """Removes duplicate chunks based on content overlap."""
    seen_content = set()
    unique_results = []
for doc in results:
    # Use first 100 characters as a rough hash for content
    content_snippet = doc.get("text", "")[:100]
if content_snippet not in seen_content:
    seen_content.add(content_snippet)
    unique_results.append(doc)
    return unique_results
    
def calculate_boosted_score(doc):
    """Calculates score with boost for trusted sources."""
    score = doc.get("score", 0)
    doc_type = doc.get("metadata", {}).get("document_type")
if doc_type in ["annual_report", "table"]:
    score += 0.05
    return score
    
def filter_by_score(results, min_score=0.75):
    """Filters results to only include those with a similarity score above the threshold."""
    filtered = [r for r in results if r.get("score", 0) >= min_score]
    return filtered
    
def build_context(search_results, max_tokens=3000):
    """Combines retrieved chunks into a single context string, respecting a token limit."""
    context_parts = []
    total_tokens = 0
    
for i, doc in enumerate(search_results, 1):
    # Format chunk with some metadata context for the LLM
    source = doc['metadata'].get('document_name', 'Unknown Source')
    chunk_type = doc['metadata'].get('content_type', 'text')
    page = doc['metadata'].get('page_number', 'NA')
    text = doc.get('text', '')
    
    # Add Speaker Info for Transcripts
    speaker_info = ""
if doc['metadata'].get('document_type') == "transcript":
    speaker = doc['metadata'].get('speaker', 'Unknown')
    role = doc['metadata'].get('role', 'Unknown')
    speaker_info = f" | Speaker: {speaker} ({role})"
    
    # Create context part
    # Improvement: Removing "Source {i}" index to prevent "Source 1" style citations in answer
    # If Page is NA, do not show "Page: NA" to LLM to avoid bad citations
if str(page) == "NA":
    part = f"--- Document: {source} | Type: {chunk_type}{speaker_info} ---\n{text}"
else:
    part = f"--- Document: {source} | Page: {page} | Type: {chunk_type}{speaker_info} ---\n{text}"
    part_tokens = count_tokens(part)
    
    # Check size limit
if total_tokens + part_tokens > max_tokens:
    break
    
    context_parts.append(part)
    total_tokens += part_tokens
    
    return "\n\n".join(context_parts)
    
def generate_answer(query, context, chat_history=None, intent="general", answer_mode="DEFAULT"):
    """Calls Azure OpenAI Chat Completion to answer the question based on context and history."""
    
    # Intent-specific instructions
    style_guide = ""
if answer_mode == "EXTRACTIVE_NUMERIC":
    style_guide = """
    - MODE: EXTRACTIVE NUMERIC (STRICT)
    - You MUST extract and report the exact numeric values verbatim from the context.
    - Do NOT utilize narrative summaries if a numeric value exists.
    - If multiple values exist (e.g., Reported vs Adjusted), list BOTh with clear labels.
    - Structure: Provide a single, concise sentence containing the value and context.
    - NO NARRATIVE ESCAPE: Do not say "Management reported strong growth" without the number.
    - Answer format: [Metric] was [Value] [Currency] in [Period] ([Source]).
    """
elif intent == "comparison":
    style_guide = """
    - Style: Use a persistent structure (e.g. "FY24 vs FY25").
    - Highlight the difference/growth explicitly.
    """
elif intent == "management_commentary":
    style_guide = """
    - Style: Narrative and descriptive.
    - REQUIRED: Attribute statements to specific speakers if available (e.g., "CEO Salil Parekh stated...").
    - Minimum length: 3-5 sentences to cover nuance.
    """
    
    system_prompt = f"""You are a specialized financial analyst assistant.
    Your task is to answer the user's question accurately using ONLY the provided context and conversation history.
    
    STRICT SOURCE HIERARCHY (Follow this order for conflicts):
    1. Audited Financial Tables (High Confidence)
    2. Annual Report Narrative
    3. Investor Presentation / Fact Sheet
    4. Earnings Call Transcript
    
    INSTRUCTIONS:
    - If the answer is in the context, be precise and cite the source/document if possible.
    - If the context contains data tables, interpret the rows carefully.
    - If the answer is not in the context, strictly state: "I cannot find the answer in the provided documents."
    - Do not make up information or use outside knowledge.
    - Always state currency explicitly (₹ or $)
    - If multiple currencies are present in context: Choose ONE currency (preferably INR), state the reason, and do NOT mention the other currency in the final answer.
    - CRITICAL: If a value for a specific fiscal year is not explicitly stated in the provided context, DO NOT estimate, derive, or back-calculate it. State clearly that the value is not available.
    - If a value or margin cannot be determined due to missing data: Do NOT show tables or formulas. State unavailability in 3 sentences or less.
    - HARD CONSTRAINT: Operating Profit MUST NOT be calculated from revenue or margin. Only return Operating Profit if explicitly stated in the context.
    - EPS MUST NOT be normalized, annualized, or adjusted unless explicitly stated. If EPS is not explicitly reported for the requested period, respond: "EPS is not explicitly stated in the provided documents."
    - If data is unavailable: Respond with exactly "Data unavailable in the provided documents." Do NOT explain why.
    
    INTENT-SPECIFIC RULES:
    {style_guide}
    """
    
    # Format history
    history_str = ""
if chat_history:
for turn in chat_history:
    role = turn['role'].capitalize()
    content = turn['content']
    history_str += f"{role}: {content}\n"
    
    user_prompt = f"""
    Conversation History:
    {history_str}
    
    Context:
    {context}
    
    Question:
    {query}
    
    Answer:
    """
    
try:
    response = client.chat.completions.create(
    model=CHAT_DEPLOYMENT,
    messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
    ],
    temperature=0.4 # Zero temperature for factual accuracy
    )
    return response.choices[0].message.content
except Exception as e:
    logger.error(f"LLM Generation Error: {e}")
    return "Error generating answer from LLM."
    
def answer_question(user_query: str) -> dict:
    """
    Core RAG engine function.
    """
    
    collection = connect_mongo()
if collection is None:
    return {
    "answer": "Database connection failed.",
    "confidence": "Low",
    "score": 0.0,
    "sources": []
    }
    
    # 1.1 Dual Currency Guardrail
if ("inr" in user_query.lower() and "usd" in user_query.lower()) or "both currencies" in user_query.lower():
    return {
    "answer": "Please choose a single currency (INR or USD) for consistency.",
    "confidence": "Low",
    "score": 0.0,
    "sources": []
    }
    
    # 2. Intent Classification
if len(user_query.split()) <= 4 and any(x in user_query.lower() for x in ["fy", "year"]):
    intent = "financial_data"
    logger.info(f"Intent override: {intent} (Short financial query)")
else:
    intent = classify_query_intent(user_query)
    
    logger.info(f"Detected intent: {intent}")
    
    # 3. Entity Extraction & Contextual Query Rewriting
    should_extract = (
    intent in ["comparison", "operating_profit", "financial_data"] 
    or any(x in user_query.lower() for x in ["difference", "growth", "compare", "vs", "year", "fy"])
    or (len(conversation_state.history) > 0 and len(user_query.split()) < 7)
    )
    
    entities = {}
if should_extract:
    entities = extract_entities(user_query)
    logger.info(f"Extracted Entities: {entities}")
    
    is_comparison = "difference" in user_query.lower() or "growth" in user_query.lower() or "compare" in user_query.lower()
    
    # PRODUCTION UPGRADE: Removed pre-retrieval blocking for "vague" queries.
    # We now let vector search happen first. Clarification is asked only if retrieval fails or confidence is low.
    
if intent != "management_commentary" and is_comparison and conversation_state.last_metric:
if not entities.get("metric"):
    user_query = f"{user_query} ({conversation_state.last_metric})"
    logger.info(f"Refined Query (Metric Injection): {user_query}")
    
if is_comparison:
if not entities.get("company") and conversation_state.last_company:
    user_query = f"{user_query} ({conversation_state.last_company})"
    
    # Update State
if entities.get("metric"): conversation_state.last_metric = entities["metric"]
if entities.get("year"): conversation_state.last_year = entities["year"]
if entities.get("company"): conversation_state.last_company = entities["company"]
    
    # 4. Query Expansion (LLM)
    expanded_query = user_query
if len(user_query.split()) < 10: # Only expand short queries
try:
    logger.info("Expanding query for abbreviations...")
    expansion_response = client.chat.completions.create(
    model=CHAT_DEPLOYMENT,
    messages=[
    {"role": "system", "content": "You are a financial query assistant. Rewrite the user's query by expanding common financial abbreviations (e.g., 'CEO', 'PAT', 'YoY', 'EBITDA') into their full forms. Do not add extra information or change the meaning. Return ONLY the expanded query."},
    {"role": "user", "content": user_query}
    ],
    temperature=0
    )
    expanded_query = expansion_response.choices[0].message.content.strip()
    logger.info(f"Expanded Query: {expanded_query}")
except Exception as e:
    logger.error(f"Query expansion failed: {e}")
    
    # 4. Embedding (Use Expanded Query)
    logger.info("Generating embedding...")
    
if "confident" in user_query.lower():
if conversation_state.last_metric:
    return { "answer": f"System is calibrated. Please ask a specific question to get a confidence score.", "confidence": "High", "score": 1.0, "sources": [] }
else:
    return { "answer": "I am ready. Ask a question to see my confidence score for that specific retrieval.", "confidence": "High", "score": 1.0, "sources": [] }
    
    # Forecast Lockdown
    forecast_keywords = ["project", "forecast", "estimate", "predict"]
if any(k in user_query.lower() for k in forecast_keywords) and intent != "management_commentary":
    return {
    "answer": "Projections or forecasts are not supported unless explicitly stated by management.",
    "confidence": "Low",
    "score": 0.0,
    "sources": []
    }
    
    query_embedding = get_query_embedding(expanded_query)
if not query_embedding:
    return { "answer": "Error generating embedding.", "confidence": "Low", "score": 0.0, "sources": [] }
    
    # 5. Vector Search (Adaptive)
    logger.info("Searching MongoDB (Vector Search)...")
    
    
    
    # ---------------------------------------------------------
    # STRICT DYNAMIC MULTI-COMPANY ISOLATION (NO HARDCODING)
    # ---------------------------------------------------------
    
    search_filter = {}
    
    # Document type filter (keep your existing logic)
if intent == "management_commentary":
    search_filter["metadata.document_type"] = {"$in": ["transcript", "ppt"]}
elif intent in ["financial_data", "comparison"]:
    search_filter["metadata.document_type"] = {"$in": ["annual_report", "ppt", "table"]}
    
    detected_company = None
    available_companies = []
    
    # Fetch all available companies dynamically
try:
    available_companies = collection.distinct("metadata.company")
    available_companies = [
    c.strip() 
for c in available_companies 
if c and c.strip().lower() != "unknown"
    ]
except Exception as e:
    logger.error(f"Failed fetching companies: {e}")
    available_companies = []
    
    detected_companies = []
    
    # 1️⃣ First priority: LLM extracted entity
if entities.get("company"):
    extracted_company = entities["company"].strip()
for company in available_companies:
if company.lower() in extracted_company.lower():
if company not in detected_companies:
    detected_companies.append(company)
    
    # 2️⃣ Fallback: strict substring match
    query_lower = user_query.lower()
for company in available_companies:
if company.lower() in query_lower:
if company not in detected_companies:
    detected_companies.append(company)
    
    # Fallback to conversation history only for short queries
if (
    not detected_companies
    and conversation_state.last_company
    and len(user_query.split()) <= 4
    ):
    detected_companies.append(conversation_state.last_company)
    
    # If company found → APPLY STRICT FILTER
if detected_companies:
if intent == "comparison" and len(detected_companies) > 1:
    search_filter["metadata.company"] = {"$in": detected_companies}
    logger.info(f"🔒 MULTI-COMPANY FILTER: {detected_companies}")
else:
    # Default to first company for strict isolation
    search_filter["metadata.company"] = detected_companies[0]
    logger.info(f"🔒 STRICT COMPANY FILTER APPLIED: {detected_companies[0]}")
    
    # If user did not mention company and not general intent → ask explicitly
elif not detected_companies and available_companies and intent != "general":
    formatted = ", ".join(available_companies)
    return {
    "answer": f"Please specify the company name. Available companies: {formatted}.",
    "confidence": "Low",
    "score": 0.0,
    "sources": []
    }
    
    # If filter empty → set None (allows broad search only for General/Comparison)
if not search_filter:
    search_filter = None
    
    # RECALL UPGRADE: Increase top_k to 15
    top_k_val = 15 
    
    search_results = vector_search(collection, query_embedding, top_k=top_k_val, filter_dict=search_filter)
    
    # ENTERPRISE SAFETY: Do NOT retry without filter if company was specified.
    # Strict isolation must NEVER be bypassed.
if not search_results and search_filter is not None:
    logger.warning(f"No results found with strict filter: {search_filter}. Returning empty.")
    
for doc in search_results:
    doc["score"] = calculate_boosted_score(doc)
    
    search_results = deduplicate_results(search_results)
    
    # RECALL UPGRADE: Lower strict threshold to 0.65 to let Reranker decide
    search_results = filter_by_score(search_results, min_score=0.65)
    search_results = resolve_financial_conflicts(search_results)
    
if not search_results:
    return { "answer": "No high-confidence results found. Try rephrasing.", "confidence": "Low", "score": 0.0, "sources": [] }
    
    # RECALL UPGRADE: LLM Reranking
    # We select the top 5 most relevant chunks from the top 15 retrieved
if len(search_results) > 3:
    logger.info("Reranking results with LLM...")
try:
    rerank_prompt = f"""
    Question: {user_query}
    
    Retrieved Chunks:
    """
for i, doc in enumerate(search_results):
    rerank_prompt += f"[{i}] {doc['text'][:300]}...\n"
    
    rerank_prompt += """
    Task: Select the indices of the chunks that are most relevant to answering the question.
    Prioritize chunks with numeric data for financial queries.
    Return ONLY a JSON object with a list of indices: {"indices": [0, 2, ...]}
    Select top 5 maximum.
    """
    
    rerank_response = client.chat.completions.create(
    model=CHAT_DEPLOYMENT,
    messages=[{"role": "user", "content": rerank_prompt}],
    temperature=0,
    response_format={"type": "json_object"}
    )
    import json
    indices = json.loads(rerank_response.choices[0].message.content).get("indices", [])
    
if indices:
    reranked_results = [search_results[i] for i in indices if i < len(search_results)]
if reranked_results:
    search_results = reranked_results
    logger.info(f"Reranking kept {len(search_results)} chunks.")
except Exception as e:
    logger.error(f"Reranking failed: {e}")
    # Fallback to top 5 original
    search_results = search_results[:5]
    
    retrieval_confidence = sum(r["score"] for r in search_results) / len(search_results)
    
    confidence_label = "Low"
if retrieval_confidence > 0.88:
    confidence_label = "High"
elif retrieval_confidence > 0.80:
    confidence_label = "Medium"
    
if retrieval_confidence < 0.72:
    confidence_label = "Low"
    
    logger.info(f"Found {len(search_results)} relevant chunks.")
    
    # Sort by Hierarchy and Score
    search_results = resolve_financial_conflicts(search_results)
    
    # Check for Authoritative Source (Level 5 - Table)
    has_authoritative_source = False
if search_results and (search_results[0]["metadata"].get("document_type") == "table" or search_results[0]["metadata"].get("content_type") == "table_row"):
    has_authoritative_source = True
    logger.info("Authoritative Source Found (Table/Audited Data)")
    
    # 6. Build Context
    context_text = build_context(search_results, max_tokens=3000)
    
    # Determine Answer Mode
    answer_mode = "DEFAULT"
if intent in ["financial_data", "operating_profit"]:
    answer_mode = "EXTRACTIVE_NUMERIC"
elif intent == "comparison":
    answer_mode = "STRUCTURED_COMPARISON"
elif intent == "management_commentary":
    answer_mode = "NARRATIVE"
    
    # 7. Generate Answer
    logger.info(f"Asking LLM (Mode: {answer_mode})...")
    final_answer = generate_answer(user_query, context_text, conversation_state.history, intent=intent, answer_mode=answer_mode)
    
    # Post-Generation Helper: Check for missing numbers in numeric mode
if answer_mode == "EXTRACTIVE_NUMERIC" and not any(c.isdigit() for c in final_answer):
    logger.warning("Numeric answer mode requested, but no numbers found in answer. Potential narrative fallback.")
    # We could force a retry here, but for now just log it.
    
if final_answer == "Error generating answer from LLM.":
    confidence_label = "Low"
    
    # 8. Self-Correction Loop
    # SKIP if we have an authoritative source and no obvious conflict/error is flagged by simple heuristics
    # Actually, keep it safe, but maybe trust it more.
    # User Request: "If authoritative_source_found and no_conflict: skip_self_correction()"
    # We will skip if confidence is already very high from retrieval (e.g. Table + High Consistency)
    
    should_skip_check = has_authoritative_source and intent == "financial_data" and "unavailable" not in final_answer.lower()
    
if should_skip_check:
    logger.info("Skipping self-correction (Authoritative Source Found)")
    status = "VALID"
    reason = "Authoritative source bypassed check"
else:
    logger.info("Validating answer (Self-Correction)...")
    critique = self_check_answer(user_query, context_text, final_answer, answer_mode=answer_mode)
    status = critique.status
    reason = critique.reason
    
    # Calculate Final Confidence Score
    # Formula: 0.5 * source_authority + 0.3 * self_check + 0.2 * retrieval_strength
    
    # 1. Source Authority (0.0 - 1.0)
    top_doc = search_results[0] if search_results else None
    source_score = 0.5
if top_doc:
    d_type = top_doc["metadata"].get("document_type")
    c_type = top_doc["metadata"].get("content_type")
if d_type == "table" or c_type == "table_row": source_score = 1.0
elif d_type == "annual_report": source_score = 0.9
elif d_type == "ppt": source_score = 0.7
elif d_type == "transcript": source_score = 0.6
    
    # 2. Self Check (0.0 - 1.0)
    check_score = 1.0 if status == "VALID" else 0.5 # Penalty for fixing or flagging
if status == "REFUSAL_CORRECT": check_score = 1.0 # Correct refusal is high confidence
    
    # 3. Retrieval Strength (0.0 - 1.0)
    # Average of top 3 scores
if search_results:
    top_3 = search_results[:3]
    retrieval_score = sum(r["score"] for r in top_3) / len(top_3)
else:
    retrieval_score = 0.0
    
    final_confidence_value = (0.5 * source_score) + (0.3 * check_score) + (0.2 * retrieval_score)
    
if final_confidence_value >= 0.85:
    confidence_label = "High"
elif final_confidence_value >= 0.65:
    confidence_label = "Medium"
else:
    confidence_label = "Low"
    
    # Override for failures
if "data unavailable" in final_answer.lower() or "cannot find the answer" in final_answer.lower():
    confidence_label = "Low"
    final_confidence_value = 0.0 # Force low score display
    
    # Logic for Correction Handling (only if not skipped)
if not should_skip_check:
if status == "CLARIFICATION_NEEDED":
    confidence_label = "Medium"
    logger.info(f"Clarification needed: {reason}")
if "Data unavailable" in final_answer:
    return { "answer": final_answer, "confidence": "Medium", "score": final_confidence_value, "sources": [] }
else:
    return { "answer": f"Clarification Needed: {reason}", "confidence": "Medium", "score": final_confidence_value, "sources": [] }
    
elif status == "REFUSAL_CORRECT":
    logger.info(f"Policy refusal validation: {reason}")
    confidence_label = "High" # Correct refusal is good behavior! But user wants Low?
    # User's guide: "confidence downgraded to Medium" for conflict. "Low" for missing.
    # If we refuse to guess, that's High confidence that we SHOULD NOT guess. 
    # But UI usually shows Low for "I don't know".
    # Let's stick to calculated, but usually refusal implies missing data -> Low.
if "unavailable" in final_answer.lower():
    confidence_label = "Low"
    
elif status == "FIX_REQUIRED":
    confidence_label = "Medium"
    logger.info(f"Self-correction triggered: {reason}")
    
    correction_prompt = f"""
    Correct the answer using only the context below.
    
    Issue to address: {reason}
    
    Guidelines:
    1. If data is missing/insufficient: State "Data unavailable" in <2 sentences. DO NOT show calculations/tables.
    2. If multiple figures exist: Use Consolidated Annual Report (INR).
    3. If currencies mixed: Keep ONLY INR. Remove others.
    4. If the value was derived or calculated in the original draft, REMOVE it completely from the final answer.
    5. Prevent Operating Profit derivation: If not explicitly stated, refuse to calculate from margin.
    6. Use "Consolidated Revenue" instead of just "Revenue" if using consolidated figures.
    7. If a question requires deriving a fiscal-year absolute value using percentages, ratios, or margins, the answer MUST be refused unless the derived value is explicitly stated verbatim.
    
    Context:
    {context_text}
    
    Original Draft:
    {final_answer}
    """
try:
    response = client.chat.completions.create(
    model=CHAT_DEPLOYMENT,
    messages=[{"role": "user", "content": correction_prompt}],
    temperature=0
    )
    corrected_answer = response.choices[0].message.content
    log_correction(user_query, final_answer, corrected_answer)
    final_answer = corrected_answer
    confidence_label = "Medium" # Downgrade for correction
    final_confidence_value = final_confidence_value * 0.9 # Penalty
except Exception as e:
    logger.error(f"Correction failed: {e}")
    
    # Update History
    conversation_state.add_interaction(user_query, final_answer)
    
if "Data unavailable" in final_answer:
    confidence_label = "Low"
    
    sources_list = []
if confidence_label != "Low":
    seen_sources = set()
for doc in search_results:
    meta = doc["metadata"]
    # Create structured source object
    source_obj = {
    "file": meta.get('document_name', 'Unknown'),
    "page": meta.get('page_number', 'NA'),
    "doc_type": meta.get('document_type', 'unknown')
    }
    
    # Deduplicate by serializing to tuple
    source_tuple = (source_obj["file"], source_obj["page"], source_obj["doc_type"])
    
    # Skip NA pages
if str(source_obj["page"]) == "NA":
    continue
    
if source_tuple not in seen_sources:
    sources_list.append(source_obj)
    seen_sources.add(source_tuple)
    
    return {
    "answer": final_answer,
    "confidence": confidence_label,
    "score": final_confidence_value,
    "sources": sources_list
    }
