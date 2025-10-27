import os
from langchain_core.output_parsers import JsonOutputParser

import numpy as np
import uuid
from typing import List, Dict

from module.LLMs import gpt
from module.embed import EmbeddingPreprocessor
from module.prompt import curator_prompt, evaluator_prompt, generator_prompt, reflector_prompt
from module.state import State, PlaybookEntry
from module.database import PlayBookVectorStore

from config.getenv import GetEnv
from utils import Logger, highlight_print
from module.node.node_utils import DeltaJsonParser, deduplicate_playbook, prune_playbook

logger = Logger(__name__)
json_parser = JsonOutputParser()
delta_parser = DeltaJsonParser()
llm = gpt()

# embedding
embedding_model = EmbeddingPreprocessor.default_embedding_model()

# vector store
playbook_vector_store = PlayBookVectorStore()
vector_store = playbook_vector_store.get_or_create_store(embedding_model)

env = GetEnv()

generator_chain = generator_prompt() | llm | json_parser
reflector_chain = reflector_prompt() | llm | json_parser
curator_chain = curator_prompt() | llm | json_parser
evaluator_chain = evaluator_prompt() | llm | json_parser

async def generator_node(state : State) -> State:
    verbose = state.get("verbose", False)

    if verbose:
        logger.debug("GENERATOR NODE")
    
    retrieved_bullets = state['retrieved_bullets']
    playbook = "\n".join(
        f"[{entry['entry_id']}] {entry['content']}" for entry in retrieved_bullets
    ) or "No related items"

    generation = await generator_chain.ainvoke({
        "retrieved_bullets" : playbook,
        "query" : state['query']
    })

    reasoning = generation.get("reasoning", "")
    solution = generation.get("solution", "")
    used_ids = generation.get("used_bullet_ids", [])

    if verbose:
        highlight_print("reasoning", 'green')
        logger.debug(f"{reasoning[:150]}...")
        highlight_print("solution", 'green')
        logger.debug(f"{solution[:150]}...")
        highlight_print("used_ids", 'green')
        logger.debug(f"{used_ids}...")
    
    return {
        "trajectory" : [reasoning],
        "solution" : solution,
        "used_bullet_ids" : used_ids
    }


async def evaluator_node(state : State) -> State:
    # rating : positive or negative
    verbose = state.get("verbose", False)

    if verbose:
        logger.debug("EVALUATOR NODE")

    query = state['query']
    solution = state['solution']

    if not solution:
        return {"feedback" : {"rating" : "negative", "comment" : "No solution has been created"}}
    
    feedback = await evaluator_chain.ainvoke({
        "query" : query,
        "solution" : solution
    })

    return {"feedback" : feedback}

async def reflector_node(state : State) -> State:
    verbose = state.get("verbose", False)

    if verbose:
        logger.debug("REFLECTOR NODE")

    used_bullets = [
        entry for entry in state['playbook']
        if entry['entry_id'] in state.get("used_bullet_ids", [])
    ]

    used_bullets_str = "\n".join(
        f"[{entry['entry_id']}] {entry['content']}" for entry in used_bullets 
    ) or "No related items"

    reflection = await reflector_chain.ainvoke({
        "query" : state['query'],
        "trajectory" : "".join(state['trajectory']),
        "used_bullets" : used_bullets_str,
        "feedback" : str(state.get("feedback", ""))
    })

    if verbose:
        highlight_print("reflection : key insights", 'green')
        logger.debug(reflection.get("key_insights"))

    return {"reflection" : reflection}

async def curator_node(state : State) -> State:
    verbose = state.get("verbose", False)

    if verbose:
        logger.debug("CURATOR NODE")
    
    playbook = '\n'.join(f"[{entry['entry_id']}] {entry['content']}" for entry in state['playbook']) or ""
    
    max_size = state.get("max_playbook_size", env.get_playbook_config['MAX_PLAYBOOK_SIZE'])
    current_size = len(state['playbook'])

    new_insights = await curator_chain.ainvoke({
        "playbook" : playbook,
        "reflection" : state.get("reflection"),
        "current_size" : current_size,
        "max_size" : max_size
    })

    operations = new_insights.get("operations", [])
    if verbose:
        highlight_print("Curator의 제안", 'green')
        logger.debug(f"{len(operations)} 개")
    
    return {"new_insights" : operations}

async def update_playbook_node(state : State) -> State:
    verbose = state.get("verbose", False)

    if verbose:
        logger.debug("UPDATE PLAYBOOK NODE")
    
    # 원본 state의 playbook과 id 복사
    original_playbook = state['playbook']
    original_ids = {entry['entry_id'] for entry in original_playbook}

    # in memory copy로 연산 수행
    updated_playbook_memory = original_playbook.copy()
    current_step = state.get("current_step", 0)

    buillet_tags = state.get("reflection", {}).get("bullet_tags", [])

    # 이 값으로 변경된 항목 (helpful/harmful) 추적
    updated_entries_map : Dict[str, PlaybookEntry] = {}

    for tag_info in buillet_tags:
            for entry in updated_playbook_memory:
                if entry['entry_id'] == tag_info['entry_id']:
                    if tag_info['tag'] == 'helpful':
                        entry['helpful_count'] += 1
                    elif tag_info['tag'] == 'harmful':
                        entry['harmful_count'] += 1
                    entry['last_used_step'] = current_step
                    updated_entries_map[entry['entry_id']] = entry
                    break
    
    # curator의 ADD 항목 처리
    newly_added_ids = set()
    for op in state.get("new_insights", []):
        if op.get("type") == "ADD":
            new_id = str(uuid.uuid4())
            content = op['content']

            entry = PlaybookEntry(
                entry_id=new_id,
                category=op.get('category', 'general'),
                content=content,
                helpful_count=1,
                harmful_count=0,
                created_step=current_step,
                last_used_step=current_step
            )

            # 메모리에 추가
            updated_playbook_memory.append(entry)

            # 벡터저장소에 추가
            playbook_vector_store.add_playbook_entry(vector_store, entry)
            newly_added_ids.add(new_id)

    # 중복 제거 및 prune
    dedup_threshold = state.get("dedup_threshold", env.get_playbook_config['DEDUP_THRESHOLD'])

    # deduplicate_playbook은 임베딩을 다시 계산하므로 embedding_model을 전달
    final_playbook_memory = deduplicate_playbook(
        updated_playbook_memory,
        dedup_threshold,
        embedding_model
    )
    max_size = state.get("max_playbook_size", env.get_playbook_config['MAX_PLAYBOOK_SIZE'])
    if len(final_playbook_memory) > max_size:
        final_playbook_memory = prune_playbook(final_playbook_memory, max_size)

    # 벡터스토어 동기화
    final_ids = {entry['entry_id'] for entry in final_playbook_memory}

    ids_to_delete = list(original_ids - final_ids)

    if ids_to_delete:
        playbook_vector_store.delete_playbook_entries(vector_store, ids_to_delete)

        if verbose:
            logger.debug(f"VS Sync: Deleted {len(ids_to_delete)} pruned entries.")
    
    ids_to_update = (updated_entries_map.keys() & final_ids)

    entries_to_update = [
        entry for entry in final_playbook_memory
        if entry['entry_id'] in ids_to_update
    ]

    for entry in entries_to_update:
        playbook_vector_store.update_playbook_metadata(vector_store, entry)

    if verbose:
        logger.debug(f"VS Sync: Updated metadata for {len(entries_to_update)} entries.")
        highlight_print(f"Playbook 업데이트 완료 : {len(final_playbook_memory)} 항목")
    
    # 최종적으로 동기화된 메모리 플레이북을 반환
    return {"playbook": final_playbook_memory, "current_step": current_step + 1}

def retriever_playbook_node(state : State) -> State:
    verbose = state.get("verbose", False)

    if verbose:
        logger.debug("PLAYBOOK RETRIEVER NODE")
    
    query = state['query']
    top_k = env.get_playbook_config['RETRIEVAL_TOP_K']
    similarity_threshold = env.get_playbook_config['RETRIEVAL_THRESHOLD']

    if not vector_store: # vector_store 객체가 로드되었는지 확인
        logger.warning("Vector store not available. Skipping retrieval.")
        return {"retrieved_bullets": []}
    
    results_with_scores = vector_store.similarity_search_with_score(
        query=query,
        k=top_k
    )

    retrieved = []

    for doc, score in results_with_scores:
        if score >= similarity_threshold:
            # generator는 dict(json) 형태로 반환함
            entry = PlaybookEntry(
                    entry_id=doc.metadata.get('entry_id', ''),
                    category=doc.metadata.get('category', 'general'),
                    content=doc.page_content,
                    helpful_count=doc.metadata.get('helpful_count', 0),
                    harmful_count=doc.metadata.get('harmful_count', 0),
                    created_step=doc.metadata.get('created_step', 0),
                    last_used_step=doc.metadata.get('last_used_step', 0)
            )
            retrieved.append(entry)

    highlight_print(f"🔍 {len(retrieved)}개의 관련 항목 검색 (Vector Store).", 'cyan')
    return {"retrieved_bullets" : retrieved}
    
    