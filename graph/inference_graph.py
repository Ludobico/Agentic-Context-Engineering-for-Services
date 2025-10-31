from langgraph.graph import StateGraph, START, END
import asyncio

from config.getenv import GetEnv
from core.state import State
from node.nodes import generator_node
from graph.graph_utils import solution_stream

env = GetEnv()


def create_inference_graph():
    builder = StateGraph(State)

    builder.add_node("generator", generator_node)

    builder.add_edge(START, 'generator')
    builder.add_edge("generator", END)

    return builder.compile()

async def main(inputs):
    inference_graph = create_inference_graph()

    full_solution = ""

    async for token in solution_stream(inference_graph, inputs):
        print(token, end="", flush=True)
        full_solution += token


if __name__ == "__main__":

    inputs = {
        "query" : "독수리 부리는 왜 노랄까?",
        "playbook" : [],
        "solution" : "",
        "verbose" : True,
        "retrieved_bullets" : [],
        "used_bullet_ids" : [],
        "trajectory" : [],
        "reflection" : {},
        "new_insights" : [],
        "feedback" : {},
        "current_step" : 0,
        "max_playbook_size" : env.get_playbook_config['MAX_PLAYBOOK_SIZE'],
        "dudup_threshold" : env.get_playbook_config['DEDUP_THRESHOLD'],
        "retrieval_threshold" : env.get_playbook_config['RETRIEVAL_THRESHOLD']
    }

    result = asyncio.run(main(inputs))

