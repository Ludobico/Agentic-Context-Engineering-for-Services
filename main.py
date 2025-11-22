import os
import asyncio
import shutil

from config.getenv import GetEnv
from graph import create_learning_graph, create_inference_graph
from graph.graph_utils import solution_stream
from core.state import State
from utils import highlight_print

env = GetEnv()

inference_graph = create_inference_graph()
learning_graph = create_learning_graph()

async def run_ace_pipeline(state : State):
    solution = ""
    captured_data = {}

    async for token in solution_stream(inference_graph, state, captured_data):
        print(token, end="", flush=True)

        solution += token
    
    state.update(captured_data)
    state['solution'] = solution

    await run_background_learning(state)
    return solution

async def run_background_learning(state_from_inference : State):
    await learning_graph.ainvoke(state_from_inference)


async def main():
    state = {
        "playbook": [], 
        "retrieved_bullets": [],
        
        # Config
        "max_playbook_size": env.get_eval_config["MAX_PLAYBOOK_SIZE"],
        "dedup_threshold": env.get_eval_config["DEDUP_THRESHOLD"],
        "retrieval_threshold": env.get_eval_config["RETRIEVAL_THRESHOLD"],
        "retrieval_topk" : env.get_eval_config['RETRIEVAL_TOP_K'],
    }
    # State 주입
    state['query'] = """
사용자가 스페인어로 질문했을 때 영어로 답변하면 안 되는 이유가 뭐야?" 또는 "다국어 지원 시 응답 언어는 어떻게 결정해야 해?
"""
    state['ground_truth'] = ""
    
    # 다른 태스크 필드는 비워줌 (충돌 방지)
    state['test_code'] = ""
    state['entry_point'] = ""

    # 실행 상태 초기화
    state["solution"] = ""
    state["retrieved_bullets"] = []
    state["used_bullet_ids"] = []
    state["trajectory"] = []
    state["reflection"] = {}
    state["new_insights"] = []
    state["feedback"] = {}
    state['verbose'] = True

    await run_ace_pipeline(state)

if __name__ == "__main__":
    asyncio.run(main())