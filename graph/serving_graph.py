from langgraph.graph import StateGraph, START, END
import asyncio
import os
import shutil

from config.getenv import GetEnv
from core.state import State
from node.nodes import generator_node, retriever_playbook_node, router_node, simple_generator_node

def decide_route(state : State):
    if state.get("router_decision") == 'simple':
        return "simple_generator"
    else:
        return "retriever"
    
def create_serving_graph():
    builder = StateGraph(State)

    # Nodes
    builder.add_node("router", router_node)
    builder.add_node("simple_generator", simple_generator_node)
    builder.add_node("retriever", retriever_playbook_node)
    builder.add_node("generator", generator_node)

    # Edges
    builder.add_edge(START, 'router')
    builder.add_conditional_edges(
        "router",
        decide_route,
        {
            "simple_generator" : "simple_generator",
            "retriever" : "retriever"
        }
    )
    # simple task
    builder.add_edge("simple_generator", END)

    # ACE task
    builder.add_edge("retriever", "generator")
    builder.add_edge("generator", END)

    return builder.compile()