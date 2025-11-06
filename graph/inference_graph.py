from langgraph.graph import StateGraph, START, END
import asyncio

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

    # ✅ 여러 번의 질문 실행 (학습 누적)
    # ✅ ADD, UPDATE, DEDUP, PRUNE을 테스트하기 위한 질문 시나리오
    queries = [
        # 1. ADD: .sort()에 대한 새로운 정보 추가
        "파이썬에서 리스트를 정렬하는 가장 기본적인 방법은 뭐야?",
        # 2. ADD: sorted()에 대한 새로운 정보 추가
        "파이썬에서 원래 리스트를 바꾸지 않고 새롭게 정렬된 리스트를 만들려면 어떻게 해?",
        # 3. UPDATE: .sort()에 reverse=True 파라미터를 추가하도록 업데이트 유도
        "파이썬 리스트를 내림차순으로 정렬하려면 어떻게 해야 해?",
        # 4. DEDUP: 1번 질문과 유사한 질문으로 중복 추가 방지 테스트
        "파이썬 리스트를 정렬하는 방법 알려줘.",
        # 5. PRUNE: 잘못된 정보(숫자/문자 혼합 정렬)를 제공하여 harmful_count를 높이고, 해당 항목 삭제 유도
        "파이썬에서 숫자와 문자가 섞인 리스트를 정렬할 수 있어?",
    ]

    for i, q in enumerate(queries, start=1):
        print(f"\n==== 🧠 STEP {i}: {q} ====\n")
        state["current_step"] = i
        state = await run_query(inference_graph, state, q)

if __name__ == "__main__":
    asyncio.run(main())