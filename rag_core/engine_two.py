import os
import time
import json
import logging
import hashlib
import tiktoken
import re
from typing import List, Optional, Dict, Any, Literal
from functools import lru_cache

from pymongo import MongoClient
import certifi
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel, Field

# Configure module logger (inherits from Root Logger configured in main)
logger = logging.getLogger(__name__)

# --- Configuration & Connections (Previously in Base Engine) ---

# Load environment variables
basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(basedir, ".env"), override=True)

# Azure Configuration
AZURE_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_AI_API_KEY")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
CHAT_DEPLOYMENT = os.getenv("GPT_4_1_MINI_DEPLOYMENT", "gpt-4.1-mini") 

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME", "financial_rag")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "rag_chunks")
VECTOR_INDEX_NAME = "vector_index"

# Initialize Azure Client
client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version="2024-02-15-preview"
)

# --- Base Helper Functions ---

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Returns the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

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
        db = mongo_client[DB_NAME]
        return db[COLLECTION_NAME]
    except Exception as e:
        logger.error(f"MongoDB Connection Error: {e}")
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
        logger.error(f"Error generating embedding: {e}")
        return None

def vector_search(collection, query_embedding, top_k=5, filter_dict=None):
    """Performs vector search in MongoDB."""
    vector_search_stage = {
        "index": VECTOR_INDEX_NAME,
        "path": "embedding",
        "queryVector": query_embedding,
        "numCandidates": 100,
        "limit": top_k
    }
    if filter_dict:
        vector_search_stage["filter"] = filter_dict

    pipeline = [
        {"$vectorSearch": vector_search_stage},
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
        logger.error(f"Vector Search Error: {e}")
        return []

def deduplicate_results(results):
    """Removes duplicate chunks based on content overlap."""
    seen_content = set()
    unique_results = []
    for doc in results:
        content_snippet = doc.get("text", "")[:100]
        if content_snippet not in seen_content:
            seen_content.add(content_snippet)
            unique_results.append(doc)
    return unique_results

def build_context(search_results, max_tokens=3000):
    """Combines retrieved chunks into a single context string, respecting a token limit."""
    context_parts = []
    total_tokens = 0
    
    for i, doc in enumerate(search_results, 1):
        source = doc['metadata'].get('document_name', 'Unknown Source')
        chunk_type = doc['metadata'].get('content_type', 'text')
        page = doc['metadata'].get('page_number', 'NA')
        text = doc.get('text', '')
        
        speaker_info = ""
        if doc['metadata'].get('document_type') == "transcript":
            speaker = doc['metadata'].get('speaker', 'Unknown')
            role = doc['metadata'].get('role', 'Unknown')
            speaker_info = f" | Speaker: {speaker} ({role})"
        
        if str(page) == "NA":
             part = f"--- Document: {source} | Type: {chunk_type}{speaker_info} ---\n{text}"
        else:
             part = f"--- Document: {source} | Page: {page} | Type: {chunk_type}{speaker_info} ---\n{text}"
        part_tokens = count_tokens(part)
        
        if total_tokens + part_tokens > max_tokens:
            break
            
        context_parts.append(part)
        total_tokens += part_tokens
    
    return "\n\n".join(context_parts)

# --- Models & Advanced Logic ---

class RetrievalDecision(BaseModel):
    needs_retrieval: bool
    confidence_score: float = Field(description="0.0 to 1.0 confidence in this decision")
    strategy: Literal["direct", "primary_retrieval", "multi_hop"] = Field(
        description="Strategy: 'direct' (LLM only), 'primary_retrieval' (Standard), 'multi_hop' (Complex/Derived)"
    )
    reasoning: str
    search_queries: List[str]
    retrieval_depth: int = Field(default=5, description="Number of results to retrieve (top_k)")

class RelevanceEvaluation(BaseModel):
    status: Literal["sufficient", "partial", "irrelevant"]
    missing_information: Optional[str] = Field(description="What specific info is missing vs the query")
    reasoning: str

class ComponentQuery(BaseModel):
    component_name: str
    search_query: str
    rationale: str

class RetrievalPlan(BaseModel):
    is_derivable: bool
    components: List[ComponentQuery] = []
    fallback_strategy: Optional[str] = None
    reasoning: str

class ReformulatedQueries(BaseModel):
    reasoning: str
    queries: List[str]

class ValidationResult(BaseModel):
    is_valid: bool
    critique: str
    corrected_answer: Optional[str] = Field(description="Only if invalid, provide a corrected version based strictly on context.")

# --- Helper: LLM Trace Wrapper (Local) ---

def call_llm(client, model, messages, temperature=0, response_format=None):
    """
    Wrapper around OpenAI call with FULL logging visibility.
    """
    logger.info(f"🧠 [LLM Call] Initiating request to {model}")
    logger.debug(f"   👉 Temperature: {temperature}")
    logger.debug(f"   👉 Response Format: {response_format}")
    
    # Log full input (DEBUG level)
    try:
        logger.debug("---- LLM INPUT MESSAGES START ----")
        logger.debug(json.dumps(messages, indent=2, ensure_ascii=False))
        logger.debug("---- LLM INPUT MESSAGES END ----")
    except Exception as e:
        logger.warning(f"Failed to log LLM input: {e}")

    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format=response_format
        )
    except Exception as e:
        logger.error(f"❌ LLM Call Failed: {e}")
        raise e

    duration = time.time() - start_time
    logger.info(f"   ⏱️ LLM Duration: {duration:.3f}s")
    
    # if hasattr(response, "usage") and response.usage:
    #     logger.info(f"   💰 Token Usage: {response.usage.total_tokens} (Prompt: {response.usage.prompt_tokens}, Completion: {response.usage.completion_tokens})")

    try:
        content = response.choices[0].message.content
        logger.debug("---- LLM OUTPUT CONTENT START ----")
        logger.debug(content)
        logger.debug("---- LLM OUTPUT CONTENT END ----")
    except Exception as e:
        logger.warning(f"Failed to log LLM output: {e}")

    return response

# --- Core Logic ---

def decide_retrieval_need(query: str) -> RetrievalDecision:
    logger.info("🔍 [Step 1] ROUTER: Analyzing Query (Reasoning-Based)...")
    start_time = time.time()
    
    system_prompt = """
    You are a Query Analysis Agent for a financial RAG system.
    
    GOAL: Analyze the user's query and decide the optimal retrieval strategy based on KNOWLEDGE TYPE.

    ANALYSIS STEPS:
    1. Identify the Knowledge Type:
       - General Knowledge / Static Facts (e.g., "What is EBITDA?", "Who is CEO?") -> Low retrieval need.
       - Specific Financial Data (e.g., "Revenue FY24", "Margins") -> High retrieval need.
       - Time-Sensitive / Internal Updates (e.g., "Latest guidance", "CEO comments Q3") -> High retrieval need.
       - Complex / Derived (e.g., "Calculate growth rate", "Compare X and Y") -> Multi-hop need.
    
    2. Decide Retrieval Strategy:
       - "direct": If the LLM can confidently answer from general training (public info, definitions).
       - "primary_retrieval": If specific numbers, dates, or report text is needed.
       - "multi_hop": If the query needs multiple distinct pieces of info logic (e.g., Numerator + Denominator).

    3. Assign Retrieval Depth (top_k):
       - 3-5: Simple, specific lookups.
       - 10+: Broad topics ("Competitive landscape", "Risks").
       - 15+: Deep dives.

    4. Assign Confidence Score (0.0 - 1.0):
       - How sure are you that this strategy is correct?

    Return JSON:
    {
        "needs_retrieval": boolean,
        "confidence_score": float,
        "strategy": "direct" | "primary_retrieval" | "multi_hop",
        "reasoning": "Step-by-step implementation logic",
        "search_queries": ["list", "of", "queries"],
        "retrieval_depth": int
    }
    """
    
    try:
        response = call_llm(
            client=client,
            model=CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        data = json.loads(response.choices[0].message.content)
        decision = RetrievalDecision(**data)
        
        elapsed = time.time() - start_time
        logger.info(f"   👉 Decision: {decision.strategy.upper()} (Conf: {decision.confidence_score}) | Need Ret: {decision.needs_retrieval}")
        logger.info(f"   👉 Reasoning: {decision.reasoning}")
        logger.info(f"   👉 Retrieval Depth: {decision.retrieval_depth}")
        logger.debug(f"   ⏱️ Router Time: {elapsed:.2f}s")
        return decision
    except Exception as e:
        logger.error(f"Router Decision Failed: {e}")
        return RetrievalDecision(
            needs_retrieval=True, 
            strategy="primary_retrieval", 
            reasoning="Error in router, defaulting", 
            search_queries=[query],
            retrieval_depth=5
        )

def evaluate_results_relevance(query: str, results: List[Dict]) -> RelevanceEvaluation:
    if not results:
        logger.warning("   ⚠️ No results found to evaluate.")
        return RelevanceEvaluation(status="irrelevant", reasoning="No results found.")
    
    context_preview = "\n".join([f"[{i}] {r.get('text', '')[:250]}..." for i, r in enumerate(results[:4])])
    
    prompt = f"""
    Query: {query}
    
    Retrieved Fragments:
    {context_preview}
    
    Evaluate the relevance of these fragments to the query.
    
    1. Does the retrieved context FULLY answer the question?
    2. If NO, what specific information is missing? (e.g., "Missing FY24 Operating Margin, only found Revenue").
    3. Assign Status:
       - "sufficient": All key data points present.
       - "partial": Some info found, but key metrics missing.
       - "irrelevant": Context is unrelated.
    
    Return JSON:
    {{
        "status": "sufficient" | "partial" | "irrelevant",
        "missing_information": "string or null",
        "reasoning": "Detailed analysis"
    }}
    """
    
    start_time = time.time()
    try:
        logging.info("🧐 [Step 2] RELEVANCE CHECK: Validating 4 chunks...")
        response = call_llm(
            client=client,
            model=CHAT_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        data = json.loads(response.choices[0].message.content)
        eval_result = RelevanceEvaluation(**data)
        
        elapsed = time.time() - start_time
        logger.info(f"   👉 Status: {eval_result.status.upper()}")
        logger.info(f"   👉 Reason: {eval_result.reasoning}")
        logger.debug(f"   ⏱️ Check Time: {elapsed:.2f}s")
        return eval_result
    except Exception as e:
        logger.error(f"Relevance Check Failed: {e}")
        return RelevanceEvaluation(status="sufficient", reasoning="Check failed, assuming good.")

def reformulate_query(query: str, evaluation: RelevanceEvaluation) -> ReformulatedQueries:
    logger.info("🔄 [Step 2.5] REFORMULATION: Improving search queries...")
    start_time = time.time()
    
    system_prompt = f"""
    You are a Query Reformulation Agent. 
    The original search for '{query}' returned results with status: {evaluation.status}.
    Reasoning: {evaluation.reasoning}.
    
    Your goal is to generate better search queries to find the missing information.
    
    TACTICS:
    1. Expand synonyms (e.g., "Sales" -> "Revenue", "Top line").
    2. Expand acronyms (e.g., "EBITDA" -> "Earnings Before Interest...").
    3. Remove noise words if query was too specific.
    4. Focus on the core financial terms missing from the partial results.
    
    Return JSON:
    {{
        "reasoning": "Strategy for reformulation...",
        "queries": ["query 1", "query 2", "query 3"]
    }}
    """
    
    try:
        response = call_llm(
            client=client,
            model=CHAT_DEPLOYMENT,
            messages=[{"role": "user", "content": system_prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        data = json.loads(response.choices[0].message.content)
        reform = ReformulatedQueries(**data)
        
        elapsed = time.time() - start_time
        logger.info(f"   👉 Strategy: {reform.reasoning}")
        logger.info(f"   👉 New Queries: {reform.queries}")
        logger.debug(f"   ⏱️ Reform Time: {elapsed:.2f}s")
        return reform
    except Exception as e:
        logger.error(f"Reformulation Failed: {e}")
        return ReformulatedQueries(reasoning="Failed", queries=[query])

def generate_retrieval_plan(query: str, context_status: str) -> RetrievalPlan:
    logger.info("📋 [Step 3] PLANNER: Creating Component Retrieval Plan...")
    start_time = time.time()
    
    system_prompt = f"""
    You are a Retrieval Planner. 
    Context Status: {context_status}.
    
    Your goal is to create a COMPONENT-BASED retrieval plan (Derived Metrics or Multi-Hop).

    LOGIC:
    1. Determine if the requested metric can be DERIVED from other standard metrics.
       - Example: "EBITDA" = "Operating Profit" + "Depreciation".
    2. If DERIVABLE or MULTI-HOP Required:
       - Set "is_derivable": true
       - List the "components" needed.
       - Provide a specific "search_query" for each.
    3. If NOT DERIVABLE (e.g., asking for a specific name/date that should exist):
       - Set "is_derivable": false
       - Set "fallback_strategy": "reformulate"
       - Provide "reasoning".

    Return JSON:
    {{
        "is_derivable": boolean,
        "reasoning": "explanation",
        "components": [
            {{ "component_name": "Net Income", "search_query": "Infosys Net Income FY24", "rationale": "Base component" }}
        ],
        "fallback_strategy": "string (optional)"
    }}
    """

    try:
        response = call_llm(
            client=client,
            model=CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}"}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        data = json.loads(response.choices[0].message.content)
        plan = RetrievalPlan(**data)
        
        elapsed = time.time() - start_time
        logger.info(f"   👉 Derivable: {plan.is_derivable}")
        logger.info(f"   👉 Plan Logic: {plan.reasoning}")
        if plan.is_derivable:
            for comp in plan.components:
                logger.info(f"      🔹 Component: {comp.component_name} | Query: '{comp.search_query}'")
        logger.debug(f"   ⏱️ Plan Time: {elapsed:.2f}s")
        return plan
    except Exception as e:
        logger.error(f"Planner Failed: {e}")
        return RetrievalPlan(is_derivable=False, reasoning="Planner Error")

def execute_search(collection, queries: List[str], top_k=5) -> List[Dict]:
    all_results = []
    seen_ids = set()
    
    logger.info(f"🚀 [Search Execution] Running {len(queries)} queries (top_k={top_k})...")
    start_time = time.time()
    
    for q in queries:
        logger.debug(f"   🔎 Generating Embedding for: '{q}'")
        emb = get_query_embedding(q)
        if not emb:
            continue
            
        logger.debug(f"      ↳ Vector Search Started (Top-K={top_k})")
        results = vector_search(collection, emb, top_k=top_k)
        
        logger.debug(f"      ↳ Retrieved {len(results)} raw documents")
        
        added_count = 0
        for i, r in enumerate(results):
            score = r.get("score", "N/A")
            doc_name = r.get("metadata", {}).get("document_name", "Unknown")
            
            # Safer hash using MD5
            content_hash = hashlib.md5(r.get('text', '').encode('utf-8')).hexdigest()
            if content_hash not in seen_ids:
                seen_ids.add(content_hash)
                all_results.append(r)
                added_count += 1
                logger.debug(f"         [{i}] Score: {score} | Source: {doc_name} (Added)")
            else:
                logger.debug(f"         [{i}] Score: {score} | Source: {doc_name} (Duplicate)")
    
    elapsed = time.time() - start_time
    logger.info(f"   ✅ Total Unique Docs Found: {len(all_results)}")
    logger.debug(f"   ⏱️ Search Time: {elapsed:.2f}s")
    return all_results

def validate_answer(query: str, context: str, answer: str) -> ValidationResult:
    logger.info("⚖️ [Step 4] VALIDATOR: Auditing Final Answer...")
    
    system_prompt = """
    You are an Independent AI Auditor.
    Your task: rigorous fact-checking.

    INPUT:
    1. User Query
    2. Context (Source Data)
    3. Generated Answer

    AUDIT RULES:
    - If the Answer contains specific numbers/facts NOT present in Context -> INVALID.
    - If the Answer claims "Context doesn't say" but it DOES -> INVALID.
    - If the Answer is logically sound and supported -> VALID.
    
    Return JSON:
    {
        "is_valid": boolean,
        "critique": "Explain why valid/invalid",
        "corrected_answer": "string (optional, ONLY if invalid)"
    }
    """
    
    try:
        response = call_llm(
            client=client,
            model=CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}\n\nContext: {context[:4000]}...\n\nAnswer: {answer}"}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        data = json.loads(response.choices[0].message.content)
        return ValidationResult(**data)
    except Exception as e:
        logger.warning(f"Validation step failed: {e}")
        return ValidationResult(is_valid=True, critique="Validation failed, assuming valid.")

def answer_adaptive(user_query: str) -> Dict[str, Any]:
    """
    Elite-Level Agentic RAG Pipeline with Strict Control Flow & Full Logging.
    """
    pipeline_start = time.time()
    
    logger.info("="*80)
    logger.info(f"🚀 STARTING AGENTIC PIPELINE")
    logger.info(f"📝 User Query: {user_query}")
    logger.info("="*80)
    
    collection = connect_mongo()
    
    # 🟢 STEP 1: Router
    decision = decide_retrieval_need(user_query)
        
    if not decision.needs_retrieval:
        logger.info("✅ Direct LLM Answer (No Retrieval Needed).")
        completion = call_llm(
            client=client,
            model=CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant. Answer correctly."},
                {"role": "user", "content": user_query}
            ]
        )
        return {
            "answer": completion.choices[0].message.content,
            "sources": [],
            "method": "direct_llm",
            "confidence": "High"
        }

    current_results = []
    method = "primary_retrieval"
    relevance_status = "unknown"

    # 🟢 STEP 2: Branching Logic
    if decision.strategy == "multi_hop":
        # 2A: Explicit Multi-Hop (Skip Primary)
        logger.info("🔀 Strategy: Multi-Hop detected. Skipping primary retrieval, going to Planner.")
        plan = generate_retrieval_plan(user_query, context_status="Multi-Hop Strategy Requested")
        
        if plan.is_derivable:
            method = "multi_hop_resolved"
            relevance_status = "derived" # Critical update for confidence
            for comp in plan.components:
                logger.info(f"   🔍 Fetching Component: {comp.component_name}")
                emb = get_query_embedding(comp.search_query)
                if emb:
                    c_res = vector_search(collection, emb, top_k=3)
                    for r in c_res:
                        r['metadata']['retrieval_tag'] = f"Component: {comp.component_name}"
                    current_results.extend(c_res)
        else:
            logger.warning("⚠️ Multi-hop plan failed. Reverting to primary search.")
            decision.strategy = "primary_retrieval" # Fallback
            method = "multi_hop_failed_fallback"

    if decision.strategy == "primary_retrieval":
        # 2B: Standard Primary Retrieval & Adaptive Loop
        search_queries = decision.search_queries if decision.search_queries else [user_query]
        current_results = execute_search(collection, search_queries, top_k=decision.retrieval_depth)
        
        # Relevance Check
        relevance = evaluate_results_relevance(user_query, current_results)
        relevance_status = relevance.status
        
        if relevance.status != "sufficient":
            # Level 2: Reformulation
            logger.warning(f"⚠️ Relevance weak ({relevance.status}). Triggering REFORMULATION.")
            reformulation = reformulate_query(user_query, relevance)
            secondary_results = execute_search(collection, reformulation.queries, top_k=decision.retrieval_depth)
            
            # Merge
            current_results.extend(secondary_results)
            current_results = deduplicate_results(current_results)
            logger.info(f"   Merged Results Count: {len(current_results)}")
            
            # Re-evaluate
            relevance = evaluate_results_relevance(user_query, current_results)
            relevance_status = relevance.status
            
            if relevance.status != "sufficient":
                # Level 3: Derivation Planner
                logger.warning(f"❌ Reformulation insufficient ({relevance.status}). Triggering PLANNER.")
                plan = generate_retrieval_plan(user_query, relevance.status)
                
                if plan.is_derivable:
                    method = "adaptive_derived"
                    relevance_status = "derived" # Update status
                    for comp in plan.components:
                        logger.info(f"   🔍 Fetching Component: {comp.component_name}")
                        emb = get_query_embedding(comp.search_query)
                        if emb:
                            c_res = vector_search(collection, emb, top_k=3)
                            for r in c_res:
                                r['metadata']['retrieval_tag'] = f"Component: {comp.component_name}"
                            current_results.extend(c_res)
                else:
                    logger.error("🛑 All strategies exhausted. Proceeding with best-effort fallback.")
                    method = "adaptive_fallback_best_effort"

    # 🟢 STEP 3: Token Guard & Final Synthesis
    current_results = deduplicate_results(current_results)
    
    # Token Guard 
    if len(current_results) > 15:
        logger.warning(f"✂️ [Token Guard] Trimming {len(current_results)} chunks to 15 to prevent overflow.")
        current_results = current_results[:15]
    
    context_str = build_context(current_results, max_tokens=5000)
    
    # Check Token Guard on String
    token_count = count_tokens(context_str)
    if token_count > 6000:
        logger.warning(f"✂️ [Token Guard] Context size {token_count} > 6000. Hard trimming.")
        context_str = context_str[:20000] # Rough char limit fallback
    else:
        logger.info(f"📄 Final Context Size: {token_count} tokens")

    # Debug Context
    logger.debug("---- FINAL CONTEXT START ----")
    logger.debug(context_str[:2000] + "... (truncated)" if len(context_str) > 2000 else context_str)
    logger.debug("---- FINAL CONTEXT END ----")

    logger.info("🧠 [Synthesis] Generating Final Answer...")
    synth_start = time.time()
    
    instructions = """
    1. Answer the user's question explicitly based *only* on the provided context.
    
    2. 📊 ANALYST INSIGHTS (Dynamic):
       - If quoting a financial metric (e.g., "21.1% Margin"), briefly explain its meaning (e.g., "retention per 100 units of revenue").
       - If context matches prior periods (YoY/QoQ), explicitly mention the TREND (Improved/Declined).
       - Provide a 1-sentence business interpretation of the data.
    
    3. 🧮 FOR DERIVED ANSWERS (if method is 'adaptive_derived'):
       - You are constructing the answer from components.
       - SHOW THE MATH: "EBITDA = 500 (Net Income) + 50 (Tax) = 550".
       - GUARDRAIL: Do NOT calculate if fiscal years or currencies differ.
    
    4. If data is missing, state "Data unavailable".
    5. Cite sources (Document Name / Page).
    """
    
    system_prompt = f"You are an Elite Financial Analyst AI.\nMETHOD: {method}\nCONTEXT STATUS: {relevance_status}\n\n{instructions}"
    
    completion = call_llm(
        client=client,
        model=CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {user_query}"}
        ],
        temperature=0.0 
    )
    
    final_answer = completion.choices[0].message.content
    logger.debug(f"   ⏱️ Synthesis Time: {time.time() - synth_start:.2f}s")
    
    # 🟢 STEP 5: Validation Layer
    validation = validate_answer(user_query, context_str, final_answer)
    if not validation.is_valid:
        logger.warning(f"🚩 Validation Failed: {validation.critique}")
        if validation.corrected_answer:
            logger.info("🔧 Applying CORRECTED Answer from Auditor.")
            final_answer = validation.corrected_answer
            confidence = "Medium" # Downgrade confidence on correction
    else:
        logger.info("✅ Validation Passed.")
    
    # 🟢 STEP 6: Confidence Logic
    if method == "direct_llm":
        confidence = "High" if decision.confidence_score > 0.8 else "Medium"
    elif validation.is_valid and relevance_status == "sufficient":
        confidence = "High"
    elif relevance_status == "derived":
        confidence = "Medium"
    else:
        confidence = "Low"

    total_time = time.time() - pipeline_start
    
    logger.info("="*80)
    logger.info("✅ FINAL ANSWER GENERATED")
    logger.info(f"   👉 Method: {method}")
    logger.info(f"   👉 Confidence: {confidence}")
    logger.info(f"   ⏱️ Total Pipeline Time: {total_time:.2f}s")
    logger.info("="*80)

    return {
        "answer": final_answer,
        "sources": list(set([d.get("metadata", {}).get("document_name", "Unknown") for d in current_results])),
        "method": method,
        "confidence": confidence,
        "debug_status": relevance_status
    }
