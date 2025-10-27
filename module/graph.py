import asyncio

from langgraph.graph import StateGraph, START, END

from config.getenv import GetEnv
from module.node.ACE import generator_node, curator_node, evaluator_node, reflector_node, update_playbook_node
from module.state import State


env = GetEnv()

def build_inference_graph():
    builder = StateGraph(State)

    builder.add_node("generator", generator_node)

    builder.add_edge(START, "generator")
    builder.add_edge("generator", END)

    return builder.compile()

if __name__ == "__main__":
    state = {
        "query" : "독수리부리는 왜 노랄까?",
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
        "dedup_threshold" : env.get_playbook_config['DEDUP_THRESHOLD'],
        "retrieval_threshold" : env.get_playbook_config['RETRIEVAL_THRESHOLD']
    }

    async def run():
        graph = build_inference_graph()

        await graph.ainvoke(state)

    asyncio.run(run())

    