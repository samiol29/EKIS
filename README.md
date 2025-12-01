# 🧠 EKIS: Enterprise Knowledge Intelligence System
### *A Dual-Architecture Capstone solving the "Black Box" problem in Enterprise Search.*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Google ADK](https://img.shields.io/badge/Google_ADK-0.4.0-4285F4?logo=google)
![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-green)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)
![Vertex AI](https://img.shields.io/badge/Cloud-Vertex_AI-yellow?logo=googlecloud)

---

## 📖 Executive Summary
**EKIS (Enterprise Knowledge Intelligence System)** addresses the critical "Information Retrieval Gap" in modern enterprises—where employees spend nearly **20% of their workweek** just hunting for existing data.

Rather than a monolithic "black box," EKIS implements **two distinct, standalone architectures** to demonstrate the separation of concerns in modern AI engineering:
1.  **System A (Local):** A deterministic, privacy-first vector search engine (The "Engine").
2.  **System B (Cloud):** A cognitive, multi-agent reasoning framework (The "Brain").

---

## 🏗️ The Dual-Architecture Approach

### **System A: Local RAG Retriever (No LLM)**
*Located locally in VS Code / Docker.*
Designed for speed, privacy, and deterministic accuracy. It treats search as a pure engineering problem, removing the unpredictability of generative AI.

* **Tech Stack:** Python, FAISS (CPU), SentenceTransformers (`all-MiniLM-L6-v2`), FastAPI.
* **Key Capability:** Ingests raw PDFs via a custom PyPDF pipeline, chunks them based on semantic boundaries, and exposes a RESTful API (`/v1/retriever/search`) for sub-millisecond retrieval.
* **Why it matters:** Provides a low-latency, air-gapped search infrastructure for sensitive internal documents.

### **System B: Multi-Agent Orchestrator (Google ADK)**
*Located on Kaggle / Vertex AI.*
Designed for reasoning, intent understanding, and synthesis. It abstracts away the database mechanics to focus on complex workflows.

* **Tech Stack:** Google Agent Development Kit (ADK), Gemini 2.5 Flash, Kaggle Kernels.
* **Key Capability:** A sequential pipeline of specialized agents (`Guardrail` -> `Retriever` -> `Writer`) that "thinks" before answering.
* **Why it matters:** Solves the "Context Switching" problem by maintaining long-term session state and handling complex, multi-step user queries.

---

## 🚀 Advanced Technical Innovations

We moved beyond standard RAG implementation to include four "Enterprise-Grade" features:

### **1. Hybrid Search "Brain"**
We don't just rely on vector similarity. System B implements a 3-stage retrieval process:
* **Semantic Search (FAISS):** Captures conceptual matches (e.g., "fiscal policy").
* **Keyword Search (BM25 Logic):** Captures exact terms (e.g., specific error codes).
* **Smart Query Expansion (HyDE):** Uses an LLM to rewrite vague user queries (e.g., *"What about the strategy?"*) into precise technical search terms before they touch the index.

### **2. Deterministic "Pass-Through" Pipeline**
To prevent infinite loops common in autonomous agents, we architected a strictly ordered **Sequential Pipeline**:
1.  **🛡️ Guardrail Agent:** Filters toxic or irrelevant queries *before* compute resources are wasted.
2.  **🔍 Retriever Agent:** Strictly instructed to output **raw text only**, ensuring no information is lost or hallucinated during the retrieval step.
3.  **🧠 Writer Agent:** Synthesizes the final answer using the raw data.

### **3. Chain-of-Thought Reasoning**
The Writer Agent is engineered to show its work. Instead of just answering, it outputs:
> **🧐 Analysis:** *[Internal Monologue about the data found]*
> **✅ Answer:** *[Final response with citations]*

This transparency builds user trust and makes the system audit-ready.

### **4. Lazy Loading Deployment**
To overcome Cloud Run/Vertex AI startup timeouts (Code 3 Errors), we implemented an aggressive **Lazy Loading Pattern**. The heavy PDF ingestion and indexing processes are deferred until the first user request, allowing the container to boot instantly and pass health checks.

---

## 📊 Evaluation & Validation

### **Visual Proof: The Knowledge Map**
We validated the system's understanding by projecting 768-dimensional document embeddings into 2D space using PCA.
![PCA Knowledge Map](![alt text](pca.png))
*Figure 1: Clusters show clear semantic separation between Policy Documents (left) and Technical Architectures (right), proving the "Hybrid Brain" effectively categorizes knowledge.*

### **Automated Compliance Testing**
We ran the ADK Evaluator against a "Golden Test Suite" to ensure process integrity.

| Test Case | Capability Tested | Verdict |
| :--- | :--- | :---: |
| **Strategy Retrieval** | Identifying key pillars in National Policy docs | ✅ **PASS** |
| **Technical Formula** | Retrieving complex LaTeX math (Self-Attention) | ✅ **PASS** |
| **Security Jailbreak** | Guardrail Agent blocking a hacking prompt | ✅ **PASS** |

---

## 💻 Quick Start

### **Running System A (Local API)**
```bash
cd system-1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# Endpoint active at http://localhost:8000/docs
```

### **Running System B (Agentic Cloud)**
Open the Jupyter Notebook in Kaggle or Vertex AI Workbench.
1. Add Secrets: Set GOOGLE_API_KEY.
2. Run Pipeline: Execute the SequentialAgent setup cells.
3. Launch UI: Run the final cell to start the Gradio Interactive Chat.

```python
# just run all the cells from top to bottom in order
# The Core Pipeline Logic
pipeline = SequentialAgent(
    name="EKIS_Orchestrator",
    sub_agents=[guard_agent, retriever_agent, writer_agent]
)
```

## **🌟 Impact Statement**
EKIS transforms the enterprise knowledge landscape. By combining the speed of System A with the reasoning of System B, we provide a roadmap to recapture the 20% of productivity lost to information hunting. It turns the "Full Day" wasted on search into a full day of innovation.

Developed as a Capstone Project for the Google 5 Day AI Agents Intensive.