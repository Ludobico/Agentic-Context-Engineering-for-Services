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
    queries = [
        "독수리 부리는 왜 노랄까?",
        "부엉이와 독수리의 시력 차이는 뭘까?",
        "독수리는 사냥을 어떻게 계획할까?",
    ]

    for i, q in enumerate(queries, start=1):
        print(f"\n==== 🧠 STEP {i}: {q} ====\n")
        state["current_step"] = i
        state = await run_query(inference_graph, state, q)

if __name__ == "__main__":
    asyncio.run(main())