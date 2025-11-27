## Agentic Context Engineering (ACE)

![ace](./static/ace.png)

**Agentic Context Engineering (ACE)** is a framework for self-improving Language Models that optimizes **context** rather than fine-tuning model weights. Proposed by Zhang et al. (2025), It addresses the limitations of existing prompt optimization methods, suas as **Brevity Bias** (loss of domain detail) and **Context Collapse** (degradation of information over repeated rewrites).

Instead of maintaining a static prompt or a compressed summary, ACE treats context as an **Evolving Playbook** - a dynamic collection of strategies, code snippets, and lessons learned. The framework operates through an agentic workflow consisting of three distinct roles:

1. **Generator** : Solves the task using the current playbook as a reference

2. **Reflector** : Analyzes the execution trajectory and feedback to identify the root causes of successes of failures

3. **Curator** : Synthesizes these insights into structured **Delta Updates** (Add/Update), ensuring the playbook grows incrementally without redundancy.

By leveraging this cycle, ACE enables agents to accumulate domain-specific knowledge and avoid repeating past mistakes, achieving state-of-the-art performance on complex reasoning benchmarks.

## Implementation Details & Key Differences

This repository implements the **Agentic Context Engineering (ACE)** framework proposed by Zhang et al. (2025), but with significant architectural enhancements designed for production stability, multi-language support, and execution reliability. Below is a detailed comparison between the original paper's theoretical framework and out practical implementation.

### 1. Architectural Evolution: From Concept to Service

While the paper describes a conceptual flow (Generator → Reflector → Curator), our implementation utilizes **LangGraph** to orchestrate a robust state machine with distinct functional nodes.

| Feature          | Paper (Theoretical)                                                   | Our Implementation                                                                                                                               |
| ---------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Workflow         | Sequential generation and reflection                                  | Cyclic State Graph: Explicit edges between `retriever` → `generator` → `evaluator` → `reflector` → `curator` → `update`                          |
| Evaluation       | Implicitly part of the Reflection step; relies on LLM self-assessment | Dedicated Evaluation Module: A standalone quality gate that runs before Reflection, supporting both deterministic code execution and LLM grading |
| Input Processing | Direct input usage                                                    | Query Intent Normalization: Inputs are rewritten to abstract specific entities into generalizable intent before retrieval.                       |

### 2. Service-Oriented Improvements

We introduced several features absent in the original paper to bridge the gap between academic research and a deployable service.

#### Cross-Lingual RAG Architecture (Canonical English Storage)

- **paper** : The benchmark assumes a mono-lingual (English) environment for both queries and context

- **Our Approach** : We implement a **Canonical English Storage** strategy to prevent knowledge fragmentation across languages
  - **input** : User query in **Any language** (e.g., Korean, Spanish, Japanese).
  - **Normalization**: The system rewrites the intent into a standardized English query for retrieval.
  - **Unified Knowledge**: The Playbook is maintained in English to serve as a central knowledge hub.
  - **Generation** : The Generator utilizes the English context but produces the final solution in the User's Target Language.

#### Structured Retrieval Optimization (Context-Action)

- **paper** : Relies on general semantic similarity for retrieval

- **Our Approach** : We enforce a strict **Context-Action** schema on the Curator
  - **Constraint** : New insights must be formulated as "When [Context]... Use [Action]..."
  - **Categorization** : Insights are strictly classified into technical categories (e.g., Code Snippets, Pitfalls, Best Practices)

#### Hybrid Execution-Based Evaluation

- **Paper**: Relies heavily on LLM-based natural language feedback

- **Our Approach** : We introduced a Hybrid Evaluator architecture capable of integrating objective signals
  - **Grounded Verification** : The Evaluator executes code in a sandbox to detect runtime errors or logic failures, providing indisputable facts (e.g., AttributeError, Test Failed) to the Reflector
  - **Objective Learning** : The Reflector uses this objective feedback to determine if a Playbook entry caused the failure. It applies the "Harmful" tag only when the Playbook explicitly provided misleading instructions, distinguishing between bad advice and simple generation errors

#### Decoupled Inference & Learning Pipeline (Planned)

- **Paper**: Inference and Adaptation are sequential, forcing the user to wait for the update process

- **Our Approach** : We are designing a split pipeline to optimize latency.
  - **Streaming Inference**: The system currently utilizes token streaming to provide immediate feedback to the user while the graph executes.
  - **Async Learning (In Progress)**: We plan to fully decouple the Evaluator → Update nodes into a background process. This will allow the system to scale learning operations independently of user traffic.

### 3. Concrete Pruning & Memory Management

The paper introduces the concept of "Grow-and-refine" but leaves the implementation abstract. Our system concretizes this logic through a rigorous Memory Management Strategy

1. **Poison Detection**: Automatically removes playbook entries where negative feedback (Harmful) outweighs positive usage (Helpful), cleansing the system of misleading strategies

2. **Capacity Control (LRU)**: Implements a hard limit on playbook size. When the limit is reached, it evicts entries based on a combination of Utility Score (Helpful count) and Recency (Least Recently Used)

3. **Semantic Deduplication**: The Curator proactively checks for semantic similarity before adding new entries, merging insights instead of creating duplicates to prevent context pollution
