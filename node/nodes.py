from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents import Document
import uuid
from datetime import datetime
import asyncio

from module.prompt import curator_prompt, evaluator_prompt, generator_prompt, reflector_prompt, query_rewrite_prompt
from node.node_utils import SolutionOnlyStreamCallback, StrictJsonOutputParser, prune_playbook, is_duplicate_entry, run_human_eval_test, run_hotpot_eval_test
from core import State, PlaybookEntry
from module.LLMs import gpt
from module.db_management import VectorStore, PlayBookDB, get_db_instance, get_vector_store_instance
from config.getenv import GetEnv
from utils import Logger, highlight_print

# TODO : inference와 learning 을 내부적으로 나누기, vector store, db 삭제하는 함수(테스트용)


env = GetEnv()
llm = gpt()
logger = Logger(__name__)
json_parser = StrictJsonOutputParser()

# CHAINS
generator_chain = generator_prompt() | llm | json_parser
evaluator_chain = evaluator_prompt() | llm | json_parser
reflector_chain = reflector_prompt() | llm | json_parser
curator_chain = curator_prompt() | llm | json_parser
rewrite_chain = query_rewrite_prompt() | llm | StrOutputParser()

# DB
vector_store = get_vector_store_instance()
embedding_model = vector_store.get_embedding_model
db = get_db_instance()

async def generator_node(state : State) -> State:
    logger.debug("GENERATOR")

    # Retrieved Playbook
    retrieved_bullets = state.get("retrieved_bullets", [])

    inputs = {
        "query" : state.get("query"),
        "retrieved_bullets" : retrieved_bullets
    }
    generation = await generator_chain.ainvoke(inputs)

    solution = generation.get("solution", "")
    rationale = generation.get("rationale", "")
    trajectory_content = f"## Rationale (Thought Process):\n{rationale}\n\n## Solution :\n{solution}"


    # [uuid-123, uuid-456] 같은 playbook의 entry_id 값
    used_bullet_ids = generation.get("used_bullet_ids", [])

    return {
        "solution" : solution,
        "trajectory" : [trajectory_content],
        "used_bullet_ids" : used_bullet_ids
    }


async def evaluator_node(state : State) -> State:
    logger.debug("EVALUATOR")

    query = state.get("query")
    solution = state.get("solution")

    # A. HumanEval (Code Execution)
    if state.get("test_code") and state.get("test_id"):
        test_code = state.get("test_code")
        test_id = state.get("test_id")
        
        is_success, message = run_human_eval_test(solution, test_code, test_id)
        rating = "positive" if is_success else "negative"

        feedback = {
            "rating": rating,
            "comment": f"Execution Result: {rating.upper()}.\nDetails: {message}"
        }

    # B. HotpotQA (Text Matching)
    elif state.get("ground_truth"):
        ground_truth = state.get("ground_truth")

        is_success, message = run_hotpot_eval_test(solution, ground_truth)
        rating = "positive" if is_success else "negative"

        feedback = {
            "rating": rating,
            "comment": f"Evaluation Result: {rating.upper()}.\nDetails: {message}"
        }

    # for normal
    else:
        inputs = {
            "query" : query,
            "solution" : solution
        }

        # rating 과 comment 형식의 feedback
        feedback = await evaluator_chain.ainvoke(inputs)

    return {
        "feedback" : feedback
    }


async def reflector_node(state : State) -> State:
    logger.debug("REFLECTOR")

    # retrieve된 모든 항목을 평가 대상으로 지정
    retrieved_bullets = state.get("retrieved_bullets", [])
    used_bullets_str = '\n'.join([f"[{entry['entry_id']}] {entry['content']}" for entry in retrieved_bullets])
    if not used_bullets_str:
        used_bullets_str = "No related items retrieved."

    query = state.get("query")
    trajectory = state.get("trajectory") # generator
    
    feedback = state.get("feedback", "") # evaluator

    inputs = {
        "query" : query,
        "trajectory" : trajectory,
        "used_bullets" : used_bullets_str,
        "feedback" : feedback
    }

    # root_cause, key_insight, bullet_tags 3개의 key 값을 가진 JSON 반환
    reflection = await reflector_chain.ainvoke(inputs)

    return {
        "reflection" : reflection
    }


async def curator_node(state : State) -> State:
    logger.debug("CURATOR")

    playbook_str = '\n'.join(f"[{entry['entry_id']}] {entry['content']}" for entry in state['playbook']) or "EMPTY PLAYBOOK"

    reflection = state.get("reflection")

    inputs = {
        "playbook" : playbook_str,
        "reflection" : reflection
    }

    # reasoning, operations 2개의 key값을 가진 JSON 반환
    new_insights = await curator_chain.ainvoke(inputs)

    operations = new_insights.get("operations", [])
    return {
        "new_insights" : operations
    }

async def update_playbook_node(state : State) -> State:
    # halpful, harmful count 누적안됨, 무조건 helpful이 1로 시작 -> 해결필요 -> curator에서 retrieved된 결과를 playbook에 전달하지 않아서 생긴 이슈였음
    logger.debug("PLAYBOOK DELTA UPDATE")

    updated_playbook = state['playbook'].copy()
    max_playbook_size = state.get("max_playbook_size")

    entries_to_save = set()

    # reflector 노드에서 반환되는 값, 여기에서 helpful과 harmful을 누적
    bullet_tags = state.get("reflection", {}).get("bullet_tags", [])

    for tag_info in bullet_tags:
        target_id = str(tag_info.get("entry_id", '')).strip()
        target_tag = tag_info.get("tag", "").lower()

        for entry in updated_playbook:
            current_id = str(entry['entry_id']).strip()

            if current_id == target_id:
                old_helpful = entry['helpful_count']

                if target_tag == 'helpful':
                    entry['helpful_count'] += 1
                    if state.get("verbose", False):
                        highlight_print(f"✅ Helpful Count UP! [{current_id[:8]}] {old_helpful}->{entry['helpful_count']}", 'green')
                elif target_tag == 'harmful':
                    entry['harmful_count'] += 1
                    if state.get("verbose", False):
                        highlight_print(f"❌ Harmful Count UP! [{current_id[:8]}]", 'red')
                
                entry['last_used_at'] = datetime.now()
                entries_to_save.add(entry['entry_id'])
                break
    
    # delta operation
    docs_to_add_to_vector_store = []
    ids_to_delete_from_vector_store = []

    # curator 노드의 operations 부분에서 각각 type, category, content로 나눠짐
    for op in state.get("new_insights", []):
        op_type = op.get("type").upper()

        if op_type == "ADD":
            new_id = str(uuid.uuid4())
            content = op['content']

            # 중복 제거
            if is_duplicate_entry(content, vector_store, embedding_model):
                logger.debug(f"Duplicate found for content : {content}. Skipping ADD")
                continue

            # expected values : strategy, code_snippet, pitfall, best_practice
            category = op.get("category", "uncategorized")

            entry = PlaybookEntry(
                entry_id=new_id,
                category=category,
                content=content,
                helpful_count=1, # curator에서 ADD로 판단하므로 결과가 helpful 하다고 판단(가정)
                harmful_count=0,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )

            # DB 저장용
            docs_to_add_to_vector_store.append(entry)
            db.add_entry(entry)
            updated_playbook.append(entry)
            


        elif op_type == "UPDATE":
            entry_id_to_update = op.get("entry_id")
            new_content = op.get("content")
            if not entry_id_to_update or not new_content:
                continue

            for entry in updated_playbook:
                if entry['entry_id'] == entry_id_to_update:
                    # id가 같지만 오래된(update가 필요한) 플레이북 삭제
                    if entry['entry_id'] in entries_to_save:
                        entries_to_save.remove(entry['entry_id'])
                    ids_to_delete_from_vector_store.append(entry['entry_id'])

                    entry['content'] = new_content
                    entry['updated_at'] = datetime.now()

                    db.add_entry(entry)
                    docs_to_add_to_vector_store.append(entry)
                    break
    # 루프가 끝난 후, 카운트만 변경되고(UPDATE 안됨) 아직 저장되지 않은 항목들 일괄 저장
    if entries_to_save:
        for entry in updated_playbook:
            if entry['entry_id'] in entries_to_save:
                db.add_entry(entry)
                ids_to_delete_from_vector_store.append(entry['entry_id'])
                docs_to_add_to_vector_store.append(entry)

    # prune
    all_entries_in_db = db.get_all_entries()
    max_playbook_size = state.get("max_playbook_size")

    _, ids_to_prune = prune_playbook(all_entries_in_db, int(max_playbook_size))

    if ids_to_prune:
        if state.get("verbose", False):
            logger.debug(f"Pruning {len(ids_to_prune)} entries...")
        
        ids_to_delete_from_vector_store.extend(ids_to_prune)

        for entry_id in ids_to_prune:
            db.delete_entry(entry_id)

    if ids_to_delete_from_vector_store:
        vector_store.delete_by_entry_ids(list(set(ids_to_delete_from_vector_store)))
    
    if docs_to_add_to_vector_store:
        docs = []
        for entry in docs_to_add_to_vector_store:
            doc = Document(
            page_content=entry['content'],
            metadata = {
                "entry_id" : entry['entry_id'],
                "category" : entry['category'],
                "helpful_count" : entry['helpful_count'],
                "harmful_count" : entry['harmful_count'],
                "created_at" : entry['created_at'],
                "updated_at" : entry['updated_at']
                    }
                )
            docs.append(doc)
        vector_store.to_disk(docs)

    return {"playbook" : updated_playbook}

async def retriever_playbook_node(state : State) -> State:
    logger.debug("PLAYBOOK RETRIEVER")

    query = state.get("query")
    rewritten_query = await rewrite_chain.ainvoke({"query" : query})
    top_k = int(state.get("retrieval_topk", env.get_playbook_config['RETRIEVAL_TOP_K']))
    threshold = float(state.get("retrieval_threshold", env.get_playbook_config['RETRIEVAL_THRESHOLD']))

    

    # if state.get("verbose", False):
    #     highlight_print(rewritten_query, 'blue')

    query_embedding = embedding_model.embed_query(rewritten_query)

    # 맨 처음 실행할때(벡터스토어가 존재하지 않을때) from_disk() 메서드를 실행하면 콜렉션을 못찾음
    vector_store_doc_count = vector_store.get_doc_count()

    # 벡터스토어에서 찾은 retrieved 결과를 playbook으로 전달해줘야 curator가 보고 판단함
    if vector_store_doc_count == 0:
        return {
            "retrieved_bullets": [],
            "playbook" : []
            }
    
    retriever = vector_store.from_disk()
    docs = retriever.similarity_search_by_vector(
        embedding=query_embedding,
        k=top_k,
        score_threshold=threshold
    )

    retrieved = []
    current_time = datetime.now()
    for doc in docs:
        meta = doc.metadata
        retrieved.append({
            "entry_id": meta.get("entry_id"),
            "category": meta.get("category"),
            "content": doc.page_content,
            "helpful_count": meta.get("helpful_count", 0),
            "harmful_count": meta.get("harmful_count", 0),
            "created_at": meta.get("created_at"),
            "updated_at": meta.get("updated_at"),
            "last_used_at" : current_time
        })
    
    if state.get("verbose", False):
        highlight_print(f"PLAYBOOK 벡터스토어에서 {len(retrieved)} 항목 검색됨", 'green')

    return {
        "retrieved_bullets" : retrieved,
        "playbook" : retrieved
        }







    