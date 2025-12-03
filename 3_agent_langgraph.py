#!/usr/bin/env python3
"""
Step 3: LangGraph Self-Correcting Agent
The CORE of the project - demonstrates agentic reasoning with feedback loops
"""

import pandas as pd
import os
from datetime import datetime
from typing import Any

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# --- UPDATED IMPORTS ---
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing_extensions import TypedDict
# --- END UPDATED IMPORTS ---


from utils import query_order_status, query_customer_details, initiate_refund

# --- UPDATED FOR LOCAL LM STUDIO ---
os.environ["OPENAI_API_KEY"] = "lm-studio"  # Placeholder (needed but ignored)
OPENROUTER_API_BASE = ""
FREE_MODEL = ""  # LM Studio uses whatever model is loaded
# -----------------------------------



# Define State
class AgentState(TypedDict):
    messages: list[BaseMessage]
    loop_count: int
    max_loops: int
    final_response: str

# Define Tools
@tool
def get_order_status(order_id: str) -> str:
    """Get the status of an order from the database"""
    return query_order_status(order_id)

@tool
def get_customer_info(customer_id: str) -> str:
    """Get customer information from the database"""
    return query_customer_details(customer_id)

@tool
def process_refund(order_id: str, reason: str) -> str:
    """Process a refund request for an order"""
    return initiate_refund(order_id, reason)

tools = [get_order_status, get_customer_info, process_refund]

class LangGraphAgent:
    def __init__(self, max_loops=3):
        # --- UPDATED LLM SETUP ---
        self.llm = ChatOpenAI(
            model=FREE_MODEL, 
            temperature=0.7,
            base_url=OPENROUTER_API_BASE,
            extra_headers={"HTTP-Referer": "http://localhost", "X-Title": "Multi-Agent System"}
        )
        self.llm_with_tools = self.llm.bind_tools(tools)
        # --- END UPDATED LLM SETUP ---
        self.max_loops = max_loops
    
    def worker_node(self, state: AgentState) -> AgentState:
        messages = state["messages"]
        response = self.llm_with_tools.invoke(messages)
        messages.append(response)
        return {**state, "messages": messages, "loop_count": state["loop_count"] + 1}
    
    def supervisor_node(self, state: AgentState) -> AgentState:
        """Supervisor Agent: Reviews the worker's response"""
        messages = state["messages"]
        last_message = messages[-1]
        
        # SIMPLIFIED PROMPT FOR SMALL MODELS
        supervisor_prompt = f"""You are a supervisor. Review this response:
        
        "{last_message.content}"

        If it answers the user's question well, say "APPROVED".
        If it is empty, wrong, or asks for more info when it shouldn't, say "REJECTED".
        
        Only say "APPROVED" or "REJECTED".
        """
        
        review = self.llm.invoke([HumanMessage(content=supervisor_prompt)])
        messages.append(review)
        
        return {
            **state,
            "messages": messages
        }

    
    def router(self, state: AgentState) -> str:
        messages = state["messages"]
        last_message = str(messages[-1].content).upper()
        if "APPROVED" in last_message and state["loop_count"] >= 1:
            return "end"
        if state["loop_count"] >= state["max_loops"]:
            return "end"
        return "worker"
    
    def build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("worker", self.worker_node)
        graph.add_node("supervisor", self.supervisor_node)
        graph.add_edge(START, "worker")
        graph.add_edge("worker", "supervisor")
        graph.add_conditional_edges("supervisor", self.router, {"worker": "worker", "end": END})
        return graph.compile()
    
    def run(self, user_query: str) -> str:
        agent = self.build_graph()
        initial_message = HumanMessage(content=f"""You are a helpful customer support agent.

Customer Query: {user_query}

Use available tools to answer accurately. Only provide information you can verify.
If something is not in the database, say so clearly.""")
        
        state = {"messages": [initial_message], "loop_count": 0, "max_loops": self.max_loops, "final_response": ""}
        
        try:
            final_state = agent.invoke(state)
            for msg in final_state["messages"]:
                if "APPROVED" in str(msg.content):
                    return str(msg.content).replace("APPROVED: ", "")
            return str(final_state["messages"][-1].content)
        except Exception as e:
            return f"Agent error: {str(e)}"

def run_agent_evaluation(test_data_file='test_data.csv'):
    if not os.path.exists(test_data_file):
        print(f"❌ {test_data_file} not found. Run: python 1_data_generation.py")
        return
    
    test_data = pd.read_csv(test_data_file)
    results = []
    print(f"🔄 Running LangGraph Agent on {len(test_data)} test cases...")
    print("=" * 80)
    agent = LangGraphAgent(max_loops=2)
    
    for idx, row in test_data.iterrows():
        test_id = row['test_id']
        query = row['query']
        response = agent.run(query)
        results.append({
            'test_id': test_id,
            'system': 'langgraph_agent',
            'query': query,
            'response': response,
            'ground_truth': row['ground_truth'],
            'timestamp': datetime.now().isoformat()
        })
        if (idx + 1) % 10 == 0:
            print(f"✅ Processed {idx + 1}/{len(test_data)} test cases")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('agent_results.csv', index=False)
    print(f"\n✅ Agent results saved: agent_results.csv")
    print(f"\nSample responses:")
    print(results_df[['test_id', 'query', 'response']].head(5).to_string())
    
    return results_df

if __name__ == "__main__":
    print("Starting LangGraph Agent Evaluation...")
    run_agent_evaluation()
