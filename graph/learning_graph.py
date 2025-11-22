from langgraph.graph import StateGraph, START, END
import asyncio
import os
import shutil

from config.getenv import GetEnv
from core.state import State
from node.nodes import generator_node, evaluator_node, reflector_node, curator_node, retriever_playbook_node, update_playbook_node

env = GetEnv()

def create_learning_graph():
    builder = StateGraph(State)

    # Nodes
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("reflector", reflector_node)
    builder.add_node("curator", curator_node)
    builder.add_node("update", update_playbook_node)

    # Edges
    builder.add_edge(START, "evaluator")
    builder.add_edge("evaluator", "reflector")
    builder.add_edge("reflector", "curator")
    builder.add_edge("curator", "update")
    builder.add_edge("update", END)

    return builder.compile()