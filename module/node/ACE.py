import os
from langchain_core.output_parsers import JsonOutputParser

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import uuid

from module.LLMs import gpt
from module.embed import EmbeddingPreprocessor
from module.prompt import curator_prompt, evaluator_prompt, generator_prompt, reflector_prompt
from module.state import State, PlaybookEntry
from module.db import DatabaseManager

from config.getenv import GetEnv
from utils import Logger, highlight_print
from module.node.node_utils import DeltaJsonParser

logger = Logger(__name__)
json_parser = JsonOutputParser()
delta_parser = DeltaJsonParser()
llm = gpt()
embedding_model = EmbeddingPreprocessor.default_embedding_model()

env = GetEnv()

generator_chain = generator_prompt() | llm | json_parser
reflector_chain = reflector_prompt() | llm | json_parser
curator_chain = curator_prompt() | llm | json_parser
evaluator_chain = evaluator_prompt() | llm | json_parser

async def retriever_playbook_node(state : State) -> State:
    verbose = state.get("verbose", False)

    if verbose:
        highlight_print("RETRIEVER NODE", 'green')
    
    db_manager = DatabaseManager()

    try:
        retriever = await db_manager.retrieve_relevant_entries(state['query'], env.get_playbook_config['RETRIEVAL_TOP_K'])
        if verbose:
            print(f"{len(retriever)} 개의 항목 검색")
        return {"retrieved_bullets" : retriever}
    finally:
        db_manager.close()

async def generator_node(state : State) -> State:
    verbose = state.get("verbose", False)

    if verbose:
        highlight_print("GENERATOR NODE", 'green')

    playbook_str = '\n'.join(f"[{e['entry_id']}] {e['content']}" for e in state['retrieved_bullets']) or "No related items"
    
    generation = await generator_chain.ainvoke({"retrieved_bullets" : playbook_str, "query" : state['query']})
    solution = generation.get("solution", "")
    used_ids = generation.get("used_bullet_ids", [])
    reasoning = generation.get("reasoning", '[No reasoning provided]')

    if verbose:
        print(f"💡 solution : {solution[:100]}...")

    return {
        "solution" : solution,
        "trajectory" : [reasoning],
        "used_bullet_ids" : used_ids
    }

async def evaluator_node(state : State) -> State:
    verbose = state.get("verbose", False)

    if verbose:
        highlight_print("EVALUATOR NODE", 'green')
    
    if not state['solution']:
        return {"feedback" : {"rating" : "negative", "comment" : "No solution generated."}}
    
    feedback = await evaluator_chain.ainvoke({"query" : state['query'], "solution" : state['solution']})

    if verbose:
        print(f"📊 평가 결과: {feedback.get('rating')}")
    
    return {"feedback" : feedback}

async def reflector_node(state : State) -> State:
    verbose = state.get("verbose", False)
    db_manager = DatabaseManager()

    if verbose:
        highlight_print("RELECTOR NODE", 'green')
    
    try:
        used_bullets = db_manager.get_playbook_by_ids(state.get("used_bullet_ids", []))
        used_bullets_str = "\n".join(f"[{e['entry_id']}] {e['content']}" for e in used_bullets) or "No items"

        reflection = await reflector_chain.ainvoke({
            "query" : state['query'],
            "trajectory" : "".join(state['trajectory']),
            "used_bullets" : used_bullets_str,
            "feedback" : str(state.get("feedback", ""))
        })

        if verbose:
            print(f"🤔 Reflection: {reflection.get('key_insight')}")
        
        return {"reflection" : reflection}
    finally:
        db_manager.close()

async def curator_node(state : State) -> State:
    verbose = state.get("verbose", False)
    db_manager = DatabaseManager()

    if verbose:
        highlight_print("CURATOR NODE", 'green')
    
    try:
        current_size = db_manager.get_playbook_size()
        # 논문에서는 playbook 이 커질수록 토큰이 너무 커지는 이슈가있었음
        # 그에 따라 curator의 insights들은 벡터스토어에서 검색으로 처리하고, curator chain의 playbook은 의미없는 값으로 채움
        insights = await curator_chain.ainvoke({"playbook" : "...",
                                                "reflection" : state.get("reflection"),
                                                "current_size" : current_size,
                                                "max_size" : env.get_playbook_config['MAX_PLAYBOOK_SIZE']})
        ops = insights.get("operations", [])

        if verbose:
            print(f"✍️ Curator 제안 operations: {len(ops)}개")
        
        return {"new_insights" : ops}
    
    finally:
        db_manager.close()

async def update_playbook_node(state : State) -> State:
    verbose = state.get("verbose", False)
    db_manager = DatabaseManager()

    if verbose:
        highlight_print("DELTA UPDATE NODE", 'green')

    try:
        step = state.get("current_step", 0)

        for tag_info in state.get("reflection", {}).get("bullet_tags", []):
            db_manager.update_entry_stats(tag_info['entry_id'], tag_info['tag'], step)
        
        for op in state.get("new_insights", []):
            if op.get("type") == "ADD":
                data = {"entry_id" : str(uuid.uuid4()),
                        "category" : op.get("category" , "general"),
                        "content" : op['content'],
                        "helpful_count": 1,
                        "created_step": step,
                        "last_used_step": step}
                await db_manager.add_new_entry(data)

                if verbose:
                    print(f"➕ 항목 추가: [{data['entry_id'][:8]}]")
        return {"current_step" : step + 1}
    finally : db_manager.close()
