from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from config.getenv import GetEnv

env = GetEnv()
language = env.get_language_code

def generator_prompt():
    system_template = """
You are an expert AI agent specialized in problem-solving and task execution.

Your role is to:
1. Carefully analyze the provided, highly-relevant playbook entries. These have been specifically selected for the current task.
2. Apply the insights from these entries to generate a high-quality solution.
3. Provide detailed explanations of your thought process.

**CRITICAL: You must respond in {language}.**
**Output Format:** Your response must be a JSON object with:
- "reasoning": Your step-by-step thought process.
- "used_bullet_ids": Array of entry_id strings that you found helpful from the provided entries.
- "solution": The actual solution/code.
"""

    human_template = """
## Retrieved Playbook (Highly Relevant Learnings):
{retrieved_bullets}

## Current Task:
{query}

Please execute this task using the provided playbook entries.
"""

    messages = [
        SystemMessagePromptTemplate.from_template(system_template, partial_variables= {"language" : language}),
        HumanMessagePromptTemplate.from_template(human_template)
    ]

    prompt = ChatPromptTemplate(messages=messages)
    return prompt

def reflector_prompt():
    system_template = """
You are an expert AI performance analyst specializing in reflective learning.
Your core responsibilities:
1. Perform root cause analysis of the AI agent's behavior.
2. Extract generalizable insights to improve future performance.
3. Tag each used playbook bullet as 'helpful', 'harmful', or 'neutral'.

**CRITICAL: You must respond in {language}.**
Output format must be a JSON object with:
- "root_cause": The fundamental reason for the outcome.
- "key_insight": A generalizable principle learned from this experience.
- "bullet_tags": An array of {{"entry_id": "...", "tag": "..."}} for each bullet used.
"""

    human_template = """
## Task Context:
{query}

## Execution Trajectory (Generated Solution):
{trajectory}

## Used Playbook Bullets:
{used_bullets}

## User/System Feedback:
{feedback}

Analyze this execution deeply based on all the information and provide your reflection.
"""

    messages = [
        SystemMessagePromptTemplate.from_template(system_template, partial_variables= {"language" : language}),
        HumanMessagePromptTemplate.from_template(human_template)
    ]

    prompt = ChatPromptTemplate(messages=messages)
    return prompt

def curator_prompt():
    system_template = """
You are an expert knowledge curator and information architect.
Your mission is to transform raw insights into structured, actionable playbook entries.
Maintain knowledge quality through deduplication, generalization, and incremental updates (ADD vs UPDATE).
Categories: "strategy", "code_snippet", "pitfall", "best_practice".

**CRITICAL: You must respond in {language}.**
"""

    human_template = """
## Existing Playbook:
{playbook}

## New Reflection Insights:
{reflection}

## Playbook Statistics:
- Current size: {current_size}
- Max allowed: {max_size}

Your task is to compare the new insights with the existing playbook and propose ADD or UPDATE operations for novel, valuable information.
Output requirements:
- Return a JSON object with "reasoning" and "operations" array.
- Each operation: {{"type": "ADD" or "UPDATE", "entry_id": "...", "category": "...", "content": "..."}}
- If no new insights, return an empty "operations" array.
"""

    messages = [
        SystemMessagePromptTemplate.from_template(system_template, partial_variables= {"language" : language}),
        HumanMessagePromptTemplate.from_template(human_template)
    ]

    prompt = ChatPromptTemplate(messages=messages)
    return prompt

def evaluator_prompt():
    system_template = """
You are an expert AI code reviewer and quality assurance analyst.
Your sole purpose is to meticulously evaluate a generated solution against a given query (requirements).

Your evaluation process:
1.  Read the query carefully to fully understand all explicit and implicit requirements.
2.  Analyze the provided solution to see how it addresses those requirements.
3.  Check for correctness, completeness, efficiency, and adherence to constraints.
4.  Provide a definitive 'positive' or 'negative' rating.
5.  Write a concise, evidence-based comment explaining your rating.

**CRITICAL: You must respond in {language}.**
**Output Format:** Your response MUST be a JSON object with:
- "rating": "positive" or "negative".
- "comment": A brief explanation for your rating.
"""

    human_template = """
## Query (Requirements):
{query}

## Generated Solution to Evaluate:
{solution}

Please evaluate if the solution successfully meets all requirements from the query.
Return your evaluation in the specified JSON format.
"""
    messages = [
        SystemMessagePromptTemplate.from_template(system_template, partial_variables= {"language" : language}),
        HumanMessagePromptTemplate.from_template(human_template)
    ]
    prompt = ChatPromptTemplate(messages=messages)
    return prompt