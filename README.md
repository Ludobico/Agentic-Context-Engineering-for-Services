## Agentic Context Engineering (ACE)

![ace](./static/ace.png)

**Agentic Context Engineering (ACE)** is a framework for self-improving Language Models that optimizes **context** rather than fine-tuning model weights. Proposed by Zhang et al. (2025), ACE addresses **Brevity Bias** and **Context Collapse** by treating context as an **Evolving Playbook**—a dynamic collection of strategies, code snippets, and lessons learned.

This repository transforms the theoretical ACE framework into a **production-ready, full-stack agentic service** featuring asynchronous learning, multi-model support, and a robust memory architecture.

---

## Key Features (Service-Oriented)

Unlike the original paper which focuses on the theoretical algorithm, this implementation is built for **real-world serving**.

### 1. Decoupled Architecture (Zero Latency Learning)

We separated **Inference** from **Learning** to maximize responsiveness:

- **Serving Graph**: Handles user queries, intelligent routing, and retrieval to provide immediate responses.
- **Learning Graph**: Runs in the background (asynchronous), analyzing execution trajectories to update the Playbook without blocking the user.

### 2. Multi-Provider LLM Support

Dynamically switch between SOTA models for different parts of the chain via UI/API:

- **OpenAI** (GPT-4o, GPT-4o-mini)
- **Anthropic** (Claude 3.5 Sonnet)
- **Google** (Gemini 1.5 Pro)

### 3. Smart Routing & Mode Switching

- **Standard Mode (Async)**: Intelligent router decides if a query is "Simple" (Direct Answer) or "Complex" (ACE Playbook). Learning happens in the background.
- **Full Debug Mode (Sync)**: Forces the full cycle (Retrieve → Generate → Evaluate → Reflect → Update) for visibility and debugging.

### 4. Robust Memory & Storage Stack

- **Vector Store (Qdrant)**: Stores semantic embeddings of Playbook entries.
- **Relational DB (SQLite)**: Manages metadata, usage statistics (Helpful/Harmful counts), and timestamps.
- **Session Memory (Redis)**: Manages multi-turn chat history with sliding window context.

---

## Architecture

This project utilizes **LangGraph** to orchestrate three distinct state graphs:

### A. Serving Graph

Optimized for speed.

1. **Router**: Classifies query (Simple vs. Complex).
2. **Simple Generator**: Handles chitchat/facts.
3. **Retriever**: Fetches relevant strategies.
4. **Generator**: Produces solution using the Playbook.

### B. Learning Graph

Optimized for quality. Runs asynchronously via FastAPI background tasks.

1. **Evaluator**: Hybrid evaluation using **Unit Tests (Code Execution)** and **LLM Logic**.
2. **Reflector**: Diagnoses root causes of success/failure.
3. **Curator**: Synthesizes insights into `ADD` or `UPDATE` operations.
4. **Update**: Applies changes to the Playbook (Pruning & Deduplication).

### C. Full Graph

Combines both for synchronous debugging and development.

---

## Implementation Details & Differences

We bridge the gap between academic research and deployable services with specific enhancements.

| Feature        | Paper (Theoretical)             | Our Implementation (Production)                                                      |
| :------------- | :------------------------------ | :----------------------------------------------------------------------------------- |
| **Workflow**   | Sequential (Generate → Reflect) | **Decoupled**: Serving Graph (Fast) + Background Learning Graph                      |
| **Routing**    | Process every query             | **Semantic Router**: Distinguishes "Chitchat" vs "Strategy Tasks"                    |
| **Memory**     | Abstract concept                | **Redis**: Persistent session history management                                     |
| **Storage**    | Single source                   | **Hybrid**: Qdrant (Vector) + SQLite (Meta) + Redis (Session)                        |
| **Evaluation** | LLM Feedback only               | **Hybrid Execution**: Sandbox Code Execution + LLM Reasoning                         |
| **Language**   | English Monolingual             | **Canonical English Storage**: Multilingual Input → English Logic → Localized Output |

### Core Logic & Advanced Mechanics

#### 1. Concrete Pruning & Memory Management

The paper introduces "Grow-and-refine" abstractly. We implemented a rigorous **Memory Management Strategy**:

- **Poison Detection**: Automatically removes entries where negative feedback (Harmful) outweighs positive usage (Helpful).
- **Capacity Control (LRU)**: Enforces `MAX_PLAYBOOK_SIZE`. Evicts entries based on **Utility Score** (Helpful count) and **Recency** (Last Used).
- **Semantic Deduplication**: The Curator checks vector similarity before adding new entries to prevent context pollution.

#### 2. Cross-Lingual RAG Architecture

To prevent knowledge fragmentation:

- **Input**: User asks in any language (e.g., Korean).
- **Process**: Internal logic (Retrieval, Reflection, Curation) operates in **English** to maintain a unified knowledge base.
- **Output**: The final response is generated in the user's target language.

#### 3. Structured Retrieval Optimization

The Curator enforces a schema (`Context-Action`) to maximize retrieval accuracy for "How-to" queries.

---

## Evaluation & Self-Improvement Analysis

We validated the effectiveness of the ACE framework using two challenging benchmarks: OpenAI HumanEval (Code Generation) and HotpotQA (Multi-hop Reasoning). The visualizations below demonstrate how the system autonomously improves its performance over time without any weight updates.

### Learning Curve & Knowledge Dynamics

The dashboards illustrate the real-time evolution of the agent. In both domains, the agent successfully accumulates knowledge and manages its memory within the defined constraints.

#### Reasoning Benchmark (HotpotQA)

![alt text](./evaluation/figures/hotpotqa_metrics.png)

#### Coding Benchmark (HumanEval)

![alt text](./evaluation/figures/human_eval_metrics.png)

**Key Observations**:

- Self-Improvement (Top): The Cumulative Accuracy (Blue) shows a steady upward trend in both benchmarks. Notably, HumanEval shows rapid adaptation in the early stages as the agent learns common coding patterns.
- Memory Management (Middle): The Playbook Size (Green) stabilizes exactly at the configured limit (e.g., 50 or 60 entries). The flat line confirms that our Pruning & LRU Logic is actively removing low-utility entries to prevent context pollution.
- Retrieval Utility (Bottom): The Hit Rate (Red) correlates with the success rate, proving that the Router and Retriever are effectively fetching relevant strategies for the task at hand.

### Impact of "Helpful" Context

Does the Playbook actually help? We analyzed the success rate difference between cases where retrieved context was marked "Helpful" versus cases where it was "Low Utility" (Neutral/Harmful).

#### HotpotQA (Reasoning)

![alt text](./evaluation/figures/hotpotqa_metrics_impact.png)

#### HumanEval (Coding)

![alt text](./evaluation/figures/human_eval_metrics_impact.png)
