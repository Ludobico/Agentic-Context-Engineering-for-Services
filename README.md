## Agentic Context Engineering (ACE)

![ace](./static/ace.png)

**Agentic Context Engineering (ACE)** is a framework for self-improving Language Models that optimizes **context** rather than fine-tuning model weights. Proposed by Zhang et al. (2025), It addresses the limitations of existing prompt optimization methods, suas as **Brevity Bias** (loss of domain detail) and **Context Collapse** (degradation of information over repeated rewrites).

Instead of maintaining a static prompt or a compressed summary, ACE treats context as an **Evolving Playbook** - a dynamic collection of strategies, code snippets, and lessons learned. The framework operates through an agentic workflow consisting of three distinct roles:

1. **Generator** : Solves the task using the current playbook as a reference

2. **Reflector** : Analyzes the execution trajectory and feedback to identify the root causes of successes of failures

3. **Curator** : Synthesizes these insights into structured **Delta Updates** (Add/Update), ensuring the playbook grows incrementally without redundancy.

By leveraging this cycle, ACE enables agents to accumulate domain-specific knowledge and avoid repeating past mistakes, achieving state-of-the-art performance on complex reasoning benchmarks.

## Implementation Details & Key Differences
