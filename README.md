# Self-Correcting Multi-Agent System

**Agentic AI framework with LangGraph, statistical A/B testing, and real-time dashboard.**

![Status](https://img.shields.io/badge/status-Production%20Ready-success.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)

## Overview

A **cyclic multi-agent system** that improves response quality through iterative refinement:

- **Worker Agent**: Generates answers and calls SQL tools
- **Supervisor Agent**: Reviews for hallucinations and errors
- **Feedback Loop**: Refines until approved (max 2 iterations)

## Key Features

✅ **Agentic Reasoning** - LangGraph state machine with supervisor-worker pattern  
✅ **SQL Integration** - Safe database queries via tool calling  
✅ **A/B Testing** - Statistical validation (t-tests, p-values, Cohen's d)  
✅ **LLM-as-a-Judge** - Automated evaluation on 50+ synthetic test cases  
✅ **Real-time Dashboard** - Streamlit metrics & visualizations  

## Quick Start

```bash
# 1. Setup
python -m venv venv
venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate  # Mac/Linux
pip install langchain langchain-google-genai langgraph pandas numpy scipy streamlit plotly

# 2. Run Pipeline
python utils.py                          # Init database
python 1_data_generation.py             # Generate test cases
python 2_baseline_rag.py                # Baseline system
python 3_agent_langgraph.py             # Multi-agent system
python 4_evaluation.py                  # LLM-as-a-Judge scoring
python 5_statistical_analysis.py        # A/B test & stats
streamlit run 6_streamlit_dashboard.py # Dashboard
```

**Using Local LLM (LM Studio):** Update base_url in scripts to `http://localhost:1232/v1`

## Results

| Metric         | Baseline | Agent | Delta  |
| -------------- | -------- | ----- | ------ |
| Accuracy       | 0.230    | 0.187 | -18.9% |
| P-Value        | -        | 0.44  | N/S    |
| Hallucinations | 12/50    | 10/50 | -20%   |

**Note:** Results depend on LLM quality. Tested with local Qwen2-1.5B.

## Project Structure

```
├── utils.py                    # Database & tools
├── 1_data_generation.py        # 50 synthetic test cases
├── 2_baseline_rag.py           # Simple RAG (control)
├── 3_agent_langgraph.py        # Self-correcting agent
├── 4_evaluation.py             # LLM-as-a-Judge
├── 5_statistical_analysis.py   # T-test, p-values
├── 6_streamlit_dashboard.py    # Real-time metrics
├── orders.db                   # SQLite (auto-created)
└── *.csv, *.json              # Results & reports
```

## Technical Stack

- **Agent Framework**: LangGraph
- **LLM Options**: Gemini API, OpenRouter, Local (LM Studio)
- **Evaluation**: LLM-as-a-Judge, scipy.stats
- **UI**: Streamlit + Plotly
- **Database**: SQLite

## Use Cases

- Customer support automation
- Benchmark agentic frameworks
- Interview portfolio project
- Production deployment template

## API Configuration

**Google Gemini (Free Tier):**
```bash
export GOOGLE_API_KEY="AIza..."
```

**Local LM Studio:**
```bash
# In script headers:
OPENROUTER_API_BASE = "http://localhost:1232/v1"
FREE_MODEL = "qwen2-1.5b-instruct"
```

## Troubleshooting

| Issue                              | Solution                                                    |
| ---------------------------------- | ----------------------------------------------------------- |
| `ModuleNotFoundError`              | Ensure venv is activated                                    |
| `API Key not found`                | Export `GOOGLE_API_KEY` or hardcode in script               |
| `Port 8501 in use`                 | `streamlit run 6_streamlit_dashboard.py --server.port 8502` |
| `evaluation_results.csv not found` | Run steps 4 before step 6                                   |

## Next Steps

- Deploy with GPT-4/Claude for better accuracy
- Scale to 500+ test cases
- Add custom evaluation metrics
- Integrate with production systems

## Author

**Harshavardhan Lankipalli**  
[LinkedIn](http://www.linkedin.com/in/harsha-vardhan-847296257) | [GitHub](https://github.com/Harshadev-24)

## License

MIT License - See LICENSE file for details.


