import os
import asyncio
import shutil

from config.getenv import GetEnv
from graph import create_learning_graph, create_inference_graph
from graph.graph_utils import solution_stream
from core.state import State

env = GetEnv()

inference_graph = create_inference_graph()
learning_graph = create_learning_graph()

async def run_ace_pipeline(state : State):
    solution = ""
    captured_data = {}

    async for token in solution_stream(inference_graph, state):
        print(token, end="", flush=True)

        solution += token
    
    inference_state = {**state, **captured_data}
    inference_state['solution'] = solution

    # [디버깅] 데이터가 잘 넘어왔나 확인 (중요!)
    retrieved_count = len(inference_state.get("retrieved_bullets", []))
    used_count = len(inference_state.get("used_bullet_ids", []))
    print(f"🔍 State Capture Check: Retrieved={retrieved_count}, Used={used_count}")
    

    return solution

async def run_background_learning(state_from_inference : State):
    await learning_graph.ainvoke(state_from_inference)


async def main():
    state = {}

    await run_ace_pipeline(state)

if __name__ == "__main__":
    asyncio.run(main())