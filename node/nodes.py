from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents import Document
import uuid
from datetime import datetime

from module.prompt import curator_prompt, evaluator_prompt, generator_prompt, reflector_prompt
from node.node_utils import SolutionOnlyStreamCallback, StrictJsonOutputParser, prune_playbook, is_duplicate_entry
from core import State, PlaybookEntry
from module.LLMs import gpt
from module.db_management import VectorStore, PlayBookDB, get_db_instance, get_vector_store_instance
from config.getenv import GetEnv
from utils import Logger, highlight_print


env = GetEnv()
llm = gpt()
logger = Logger(__name__)
json_parser = StrictJsonOutputParser()

# CHAINS
generator_chain = generator_prompt() | llm | json_parser
evaluator_chain = evaluator_prompt() | llm | json_parser
reflector_chain = reflector_prompt() | llm | json_parser
curator_chain = curator_prompt() | llm | json_parser

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

    # [uuid-123, uuid-456] 같은 playbook의 entry_id 값
    used_id = generation.get("used_bullet_ids", [])

    return {
        "solution" : solution,
        "trajectory" : [solution],
        "used_bullet_ids" : used_id
    }


async def evaluator_node(state : State) -> State:
    logger.debug("EVALUATOR")

    query = state.get("query")
    solution = state.get("solution")

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

    used_bullets = [entry for entry in state['playbook'] if entry['entry_id'] in state.get("used_bullet_ids", [])]
    used_bullets_str = [f"[{entry['entry_id']}] {entry['content']}" for entry in used_bullets] or "No related items"

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

    highlight_print(reflection, 'magenta')

    return {
        "reflection" : reflection
    }


async def curator_node(state : State) -> State:
    logger.debug("CURATOR")

    playbook_str = '\n'.join(f"[{entry['entry_id']}] {entry['content']}" for entry in state['playbook']) or ""

    reflection = state.get("reflection")

    inputs = {
        "playbook" : playbook_str,
        "reflection" : reflection
    }

    # reasoning, operations 2개의 key값을 가진 JSON 반환
    new_insights = await curator_chain.ainvoke(inputs)

    print(new_insights)

    operations = new_insights.get("operations", [])
    return {
        "new_insights" : operations
    }

async def update_playbook_node(state : State) -> State:
    logger.debug("PLAYBOOK DELTA UPDATE")

    updated_playbook = state['playbook'].copy()
    current_step = state.get("current_step", 0)

    # reflector 노드에서 반환되는 값
    bullet_tags = state.get("reflection", {}).get("bullet_tags", [])
    for tag_info in bullet_tags:
        for entry in updated_playbook:
            if entry['entry_id'] == tag_info['entry_id']:
                if tag_info['tag'] == 'helpful' : entry['helpful_count'] += 1
                elif tag_info['tag'] == 'harmful' : entry['harmful_count'] += 1
                entry['last_used_at'] = current_step
                break
    
    # delta operation
    docs_to_add_to_vector_store = []
    ids_to_delete_from_vector_store = []

    # curator 노드의 operations 부분에서 각각 type, category, content로 나눠짐
    for op in state.get("new_insights", []):
        if op.get("type").upper() == "ADD":
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
                # curator에서 ADD로 판단하므로 결과가 helpful 하다고 판단(가정)
                helpful_count=1,
                harmful_count=0,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )

            # vector store 저장용
            doc = Document(
                page_content=entry['content'],
                metadata = {
                    "entry_id" : entry['entry_id'],
                    "category" : entry['category'],
                    "helpful_count" : entry['helpful_count'],
                    "harmful_count" : entry['harmful_count'],
                    "created_at" : entry['created_at'].isoformat(),
                    "updated_at" : entry['updated_at'].isoformat()
                }
            )
            vector_store.to_disk([doc])

            # DB 저장용
            db.add_entry(entry)
            updated_playbook.append(entry)
            docs_to_add_to_vector_store.append(doc)


        elif op.get("type").upper() == "UPDATE":
            entry_id_to_update = op.get("entry_id")
            new_content = op.get("content")
            if not entry_id_to_update or not new_content:
                continue

            for entry in updated_playbook:
                if entry['entry_id'] == entry_id_to_update:
                    # id가 같지만 오래된(update가 필요한) 플레이북 삭제
                    ids_to_delete_from_vector_store.append(entry['entry_id'])

                    entry['content'] = new_content
                    entry['updated_at'] = datetime.now()

                    db.add_entry(entry)
                    

                    doc = Document(
                        page_content=entry['content'],
                        metadata = {
                        "entry_id" : entry['entry_id'],
                        "category" : entry['category'],
                        "helpful_count" : entry['helpful_count'],
                        "harmful_count" : entry['harmful_count'],
                        "created_at" : entry['created_at'].isoformat(),
                        "updated_at" : entry['updated_at'].isoformat()
                        }
                    )
                    docs_to_add_to_vector_store.append(doc)
                    break
    
    updated_playbook, ids_to_prune = prune_playbook(updated_playbook)

    if ids_to_prune:
        logger.debug(f"Pruning {len(ids_to_prune)} entries")
        for entry_id in ids_to_prune:
            db.delete_entry(entry_id)
        ids_to_delete_from_vector_store.extend(ids_to_prune)

    if ids_to_delete_from_vector_store:
        vector_store.delete_by_entry_ids(list(set(ids_to_delete_from_vector_store)))
    
    if docs_to_add_to_vector_store:
        vector_store.to_disk(docs_to_add_to_vector_store)

    return {"playbook" : updated_playbook}

async def retriever_playbook_node(state : State) -> State:
    logger.debug("PLAYBOOK RETRIEVER")

    query = state.get("query")
    top_k = int(env.get_playbook_config['RETRIEVAL_TOP_K'])
    threshold = float(state.get("retrieval_threshold", env.get_playbook_config['RETRIEVAL_THRESHOLD']))

    query_embedding = embedding_model.embed_query(query)

    # 맨 처음 실행할때(벡터스토어가 존재하지 않을때) from_disk() 메서드를 실행하면 콜렉션을 못찾음
    vector_store_doc_count = vector_store.get_doc_count()

    if vector_store_doc_count == 0:
        return {"retrieved_bullets": []}
    
    retriever = vector_store.from_disk()
    docs = retriever.similarity_search_by_vector(
        embedding=query_embedding,
        k=top_k,
        score_threshold=threshold
    )

    retrieved = []
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
        })
    
    
    
    highlight_print(f"PLAYBOOK 벡터스토어에서 {len(retrieved)} 항목 검색됨", 'green')
    return {"retrieved_bullets" : retrieved}







    