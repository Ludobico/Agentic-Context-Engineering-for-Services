from langgraph.graph import StateGraph, START, END
import asyncio
import os
import shutil

from config.getenv import GetEnv
from core.state import State
from node.nodes import generator_node, evaluator_node, reflector_node, curator_node, retriever_playbook_node, update_playbook_node
from graph.graph_utils import solution_stream

env = GetEnv()


def create_inference_graph():
    builder = StateGraph(State)

    builder.add_node("retriever", retriever_playbook_node)
    builder.add_node("generator", generator_node)
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("reflector", reflector_node)
    builder.add_node("curator", curator_node)
    builder.add_node("update", update_playbook_node)

    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "generator")
    builder.add_edge("generator", "evaluator")
    builder.add_edge("evaluator", "reflector")
    builder.add_edge("reflector", "curator")
    builder.add_edge("curator", "update")
    builder.add_edge("update", END)

    return builder.compile()

async def run_query(inference_graph, state, query: str):
    state["query"] = query

    async for token in solution_stream(inference_graph, state):
        print(token, end="", flush=True)

    print("\n")
    return state

def delete_db():
    db_dir = env.get_db_dir
    vector_db_dir = env.get_vector_store_dir

    for path in [db_dir, vector_db_dir]:
        if os.path.exists(path):
            shutil.rmtree(path)
    return None


async def main():
    inference_graph = create_inference_graph()

    # 초기 state
    state = {
        "query": "",
        "playbook": [],   # ✅ 한 번만 초기화하고 이후 계속 갱신됨
        "solution": "",
        "verbose": True,
        "retrieved_bullets": [],
        "used_bullet_ids": [],
        "trajectory": [],
        "reflection": {},
        "new_insights": [],
        "feedback": {},
        "current_step": 0,
        "max_playbook_size": env.get_playbook_config["MAX_PLAYBOOK_SIZE"],
        "dedup_threshold": env.get_playbook_config["DEDUP_THRESHOLD"],
        "retrieval_threshold": env.get_playbook_config["RETRIEVAL_THRESHOLD"],
    }

    queries = [
            # 1. [Create] 초기 지식 생성
            # 기대: Playbook에 새로운 항목(Entry)이 ADD 됨 (helpful=1)
            "파이썬에서 현재 날짜와 시간을 구하는 가장 기본적인 코드를 알려줘.",

            # 2. [Helpful Accumulate] 유용한 지식 재사용 및 카운트 증가
            # 기대: 1번에서 만든 Entry가 검색(Retrieved)되고, 답변에 사용됨 -> helpful_count가 2로 증가
            "방금 알려준 방법으로 오늘 날짜만 'YYYY-MM-DD' 형식으로 출력하려면 어떻게 수정해?",

            # 3. [Refine/Update] 지식의 한계 발견 및 업데이트 (또는 Harmful 판정 유도)
            # 의도: 단순 datetime.now()는 타임존 정보가 없다는 문제를 제기
            # 기대: 기존 Entry의 내용이 수정되거나(UPDATE), 새로운 타임존 관련 Entry가 추가됨. 
            # (만약 이전 답변이 틀렸다고 평가되면 harmful_count 증가 가능)
            "알려준 코드로 시간을 구했는데, 이게 한국 시간인지 UTC인지 모르겠어. 명시적으로 'Asia/Seoul' 타임존을 적용하려면 어떻게 해?",

            # 4. [Deduplication] 중복 방지 테스트
            # 의도: 3번과 거의 동일한 의도의 질문을 던짐
            # 기대: Curator가 "이미 타임존 관련 전략이 있다"고 판단하여 불필요한 ADD 연산을 수행하지 않고(Duplicate found), 기존 카운트만 올리거나 무시해야 함.
            "파이썬 datetime 객체를 특정 타임존(예: 서울)으로 설정하는 방법을 다시 설명해줘.",

            # 5. [Create & Prune] 용량 초과 유도 및 가지치기
            # 의도: 날짜와 상관없는 새로운 주제 2개를 연속으로 던져 Playbook 용량(예: max=3)을 채움
            # 기대: 새로운 Entry가 추가되면서, 가장 오래되었거나(LRU) helpful 카운트가 낮은 1번(기본 날짜) 또는 관련 없는 항목이 삭제(DELETE)됨.
            "파이썬에서 딕셔너리 두 개를 하나로 합치는(merge) 최신 문법은 뭐야?",
            
            # 6. [Prune 확인] 가지치기 확정
            "파이썬 리스트에서 중복을 제거하면서 순서를 유지하는 방법은?"
        ]

    for i, q in enumerate(queries, start=1):
        print(f"\n==== 🧠 STEP {i}: {q} ====\n")
        state["current_step"] = i
        state = await run_query(inference_graph, state, q)
        

if __name__ == "__main__":
    asyncio.run(main())