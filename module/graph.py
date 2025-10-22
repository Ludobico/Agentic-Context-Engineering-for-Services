from langgraph.graph import StateGraph, START, END

from module.state import State
from module.node.ACE import curator_node, generator_node, reflector_node, update_playbook_node, retriever_playbook_node

def should_retrieve_gate(state : State) -> str:
    return "generate_with_context" if state.get("retrieved_bullets") else "generate_direct"

def build_inference_graph():
    builder = StateGraph(State)
    builder.add_node("retriever", retriever_playbook_node)
    builder.add_node("generator", generator_node)

    builder.add_edge(START, "retriever")
    builder.add_conditional_edges("retriever", should_retrieve_gate, {
        "generate_with_context": "generator", "generate_direct": "generator"
    })
    builder.add_edge("generator", END)

    return builder.compile()

def build_learning_graph():
    builder = StateGraph(State)
    builder.add_node("reflector", reflector_node)
    builder.add_node("curator", curator_node)
    builder.add_node("update_playbook", update_playbook_node)

    builder.add_edge(START, "reflector")
    builder.add_edge("reflector", "curator")
    builder.add_edge("curator", "update_playbook")
    builder.add_edge("update_playbook", END)
    return builder.compile()

inference_graph = build_inference_graph()
learning_graph = build_learning_graph()

