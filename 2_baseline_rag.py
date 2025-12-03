#!/usr/bin/env python3
"""
Step 2: Baseline RAG System
Simple prompt + retrieval without reasoning loops
This is what we're comparing against
"""

import pandas as pd
import os
from datetime import datetime

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# --- UPDATED IMPORTS ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
# --- END UPDATED IMPORTS ---

from utils import query_order_status, query_customer_details

# --- UPDATED API KEY & MODEL ---
# Use your OpenRouter API Key
os.environ["OPENAI_API_KEY"] = ""

# Use OpenRouter endpoint
OPENROUTER_API_BASE = ""
FREE_MODEL = ""
# --- END UPDATED API KEY & MODEL ---


def baseline_rag_response(query):
    """
    Simple RAG: Query -> Retrieve Context -> Generate Answer (ONE PASS)
    No reflection, no looping, no self-correction
    """
    # Step 1: Retrieve context from database
    context = ""
    if "order" in query.lower():
        for token in query.split():
            if token.startswith("ORD"):
                context += query_order_status(token)
                break
    if "customer" in query.lower():
        for token in query.split():
            if token.startswith("CUST"):
                context += query_customer_details(token)
                break
    if not context:
        context = "No specific order/customer information retrieved."

    # Step 2: Generate response with OpenRouter model
    try:
        # --- UPDATED LLM SETUP ---
        llm = ChatOpenAI(
            model=FREE_MODEL,
            temperature=0.7,
            base_url=OPENROUTER_API_BASE,
            extra_headers={"HTTP-Referer": "http://localhost", "X-Title": "Multi-Agent System"}
        )
        # --- END UPDATED LLM SETUP ---

        prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""You are a helpful customer support agent. 
            
Customer Query: {query}

Retrieved Context: {context}

Provide a direct answer to the customer's query based on the context. 
If you're not sure, say so. DO NOT make up information.

Answer:"""
        )
        
        chain = prompt | llm
        response = chain.invoke({"query": query, "context": context})
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

def run_baseline_evaluation(test_data_file='test_data.csv'):
    """Run baseline system on all test cases"""
    if not os.path.exists(test_data_file):
        print(f"❌ {test_data_file} not found. Run: python 1_data_generation.py")
        return
    
    test_data = pd.read_csv(test_data_file)
    results = []
    
    print(f"🔄 Running Baseline RAG on {len(test_data)} test cases...")
    print("=" * 80)
    
    for idx, row in test_data.iterrows():
        test_id = row['test_id']
        query = row['query']
        response = baseline_rag_response(query)
        results.append({
            'test_id': test_id,
            'system': 'baseline_rag',
            'query': query,
            'response': response,
            'ground_truth': row['ground_truth'],
            'timestamp': datetime.now().isoformat()
        })
        if (idx + 1) % 10 == 0:
            print(f"✅ Processed {idx + 1}/{len(test_data)} test cases")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('baseline_results.csv', index=False)
    print(f"\n✅ Baseline results saved: baseline_results.csv")
    print(f"\nSample responses:")
    print(results_df[['test_id', 'query', 'response']].head(5).to_string())
    
    return results_df

if __name__ == "__main__":
    print("Starting Baseline RAG Evaluation...")
    run_baseline_evaluation()

