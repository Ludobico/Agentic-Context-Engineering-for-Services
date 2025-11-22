from langgraph.graph import StateGraph, START, END
import asyncio
import os
import shutil

from config.getenv import GetEnv
from core.state import State
from node.nodes import generator_node, evaluator_node, reflector_node, curator_node, retriever_playbook_node, update_playbook_node

def create_inference_graph():
    builder = StateGraph(State)

    # Nodes
    builder.add_node("retriever", retriever_playbook_node)
    builder.add_node("generator", generator_node)

    # Edges
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "generator")
    builder.add_edge("generator", END) 

    return builder.compile()