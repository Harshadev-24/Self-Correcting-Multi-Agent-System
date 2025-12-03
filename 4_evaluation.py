#!/usr/bin/env python3
"""
Step 4: LLM-as-a-Judge Evaluation
Uses OpenRouter to score both baseline and agent responses
Calculates Precision, Recall, F1-score for each system
"""

import pandas as pd
import os
from datetime import datetime

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# --- UPDATED IMPORTS ---
from langchain_openai import ChatOpenAI
# --- END UPDATED IMPORTS ---


# --- UPDATED FOR LOCAL LM STUDIO ---
os.environ["OPENAI_API_KEY"] = "lm-studio"  # Placeholder (needed but ignored)
OPENROUTER_API_BASE = ""
JUDGE_MODEL = ""  # LM Studio uses whatever model is loaded
# -----------------------------------


class LLMJudge:
    def __init__(self):
        # --- UPDATED LLM SETUP ---
        self.llm = ChatOpenAI(
            model=JUDGE_MODEL, 
            temperature=0,
            base_url=OPENROUTER_API_BASE,
            extra_headers={"HTTP-Referer": "http://localhost", "X-Title": "Multi-Agent System"}
        )
        # --- END UPDATED LLM SETUP ---
    
    def evaluate_response(self, query: str, response: str, ground_truth: str) -> dict:
        evaluation_prompt = f"""You are an expert evaluator. Score the assistant's response.

Customer Query: {query}

Ground Truth / Expected Answer: {ground_truth}

Assistant's Response: {response}

Evaluate on:
1. Accuracy: Does it match the ground truth?
2. Hallucinations: Did it make up information?
3. Completeness: Does it fully answer the query?

Respond in JSON format:
{{
    "score": 0.0-1.0,
    "accuracy": "high/medium/low",
    "hallucinated": true/false,
    "reasoning": "brief explanation"
}}

Only return the JSON, no other text."""
        
        try:
            result = self.llm.invoke(evaluation_prompt)
            import json
            json_str = result.content
            try:
                eval_result = json.loads(json_str)
            except:
                eval_result = {"score": 0.5, "accuracy": "unknown", "hallucinated": False, "reasoning": json_str[:100]}
            return eval_result
        except Exception as e:
            return {"score": 0.0, "accuracy": "error", "hallucinated": True, "reasoning": str(e)}

def calculate_metrics(results: pd.DataFrame) -> dict:
    correct = (results['score'] > 0.7).sum()
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    hallucinations = results['hallucinated'].sum()
    precision = (correct - hallucinations) / correct if correct > 0 else 0
    recall = correct / total if total > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, "hallucinations": hallucinations, "total_evaluations": total}

def run_evaluation(baseline_file='baseline_results.csv', agent_file='agent_results.csv'):
    if not os.path.exists(baseline_file) or not os.path.exists(agent_file):
        print("❌ Missing results files. Run `2_baseline_rag.py` and `3_agent_langgraph.py`")
        return
    
    baseline_results = pd.read_csv(baseline_file)
    agent_results = pd.read_csv(agent_file)
    judge = LLMJudge()
    all_evaluated = []
    
    print(f"🔄 Evaluating {len(baseline_results) + len(agent_results)} responses...")
    print("=" * 80)
    
    # Evaluate baseline
    print("Evaluating Baseline RAG System...")
    for idx, row in baseline_results.iterrows():
        eval_result = judge.evaluate_response(row['query'], row['response'], row['ground_truth'])
        all_evaluated.append({**row.to_dict(), **eval_result})
        if (idx + 1) % 10 == 0:
            print(f"  ✅ Evaluated {idx + 1}/{len(baseline_results)}")
    
    # Evaluate agent
    print("\nEvaluating LangGraph Agent System...")
    for idx, row in agent_results.iterrows():
        eval_result = judge.evaluate_response(row['query'], row['response'], row['ground_truth'])
        all_evaluated.append({**row.to_dict(), **eval_result})
        if (idx + 1) % 10 == 0:
            print(f"  ✅ Evaluated {idx + 1}/{len(agent_results)}")
    
    eval_df = pd.DataFrame(all_evaluated)
    eval_df.to_csv('evaluation_results.csv', index=False)
    
    baseline_metrics = calculate_metrics(eval_df[eval_df['system'] == 'baseline_rag'])
    agent_metrics = calculate_metrics(eval_df[eval_df['system'] == 'langgraph_agent'])
    
    print("\n" + "=" * 80 + "\n📊 EVALUATION RESULTS\n" + "=" * 80)
    print(f"\n🔴 BASELINE RAG SYSTEM:\n   Accuracy: {baseline_metrics['accuracy']:.2%}\n   Precision: {baseline_metrics['precision']:.2%}\n   Recall: {baseline_metrics['recall']:.2%}\n   F1-Score: {baseline_metrics['f1_score']:.2%}\n   Hallucinations: {baseline_metrics['hallucinations']}/{baseline_metrics['total_evaluations']}")
    print(f"\n🟢 LANGGRAPH AGENT SYSTEM:\n   Accuracy: {agent_metrics['accuracy']:.2%}\n   Precision: {agent_metrics['precision']:.2%}\n   Recall: {agent_metrics['recall']:.2%}\n   F1-Score: {agent_metrics['f1_score']:.2%}\n   Hallucinations: {agent_metrics['hallucinations']}/{agent_metrics['total_evaluations']}")
    
    accuracy_improvement = ((agent_metrics['accuracy'] - baseline_metrics['accuracy']) / baseline_metrics['accuracy'] * 100) if baseline_metrics['accuracy'] > 0 else 0
    print(f"\n📈 IMPROVEMENT:\n   Accuracy Gain: {accuracy_improvement:+.1f}%\n   Hallucination Reduction: {baseline_metrics['hallucinations'] - agent_metrics['hallucinations']} fewer hallucinations")
    print(f"\n✅ Full evaluation saved: evaluation_results.csv")
    
if __name__ == "__main__":
    print("Starting LLM-as-a-Judge Evaluation...")
    run_evaluation()

