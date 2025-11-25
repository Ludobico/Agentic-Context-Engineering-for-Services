import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import asyncio
import json

from utils import Logger
from core import State, ChatRequest
from graph import create_serving_graph, create_learning_graph
from graph.graph_utils import solution_stream
from module.db_management import get_db_instance, get_vector_store_instance
from config.getenv import GetEnv
from utils import highlight_print

env = GetEnv()
logger = Logger(__name__)

serving_graph = create_serving_graph()
learning_graph = create_learning_graph()

app = FastAPI(title="ACE Framework API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def run_background_learning(state : State):
    await learning_graph.ainvoke(state)

@app.post("/chat/stream")
async def chat_stream(request : ChatRequest):
    initial_state = {
        "query" : request.query,
        "playbook" : [],
        "solution" : "",
        "verbose" : False,
        "router_decision" : "",
        "llm_provider" : request.llm_provider,
        "llm_model" : request.llm_model,
        "retrieved_bullets" : [],
        "used_bullet_ids" : [],
        "trajectory" : [],
        "reflection" : {},
        "new_insights" : [],
        "feedback" : {},
        "max_playbook_size": env.get_playbook_config["MAX_PLAYBOOK_SIZE"],
        "dedup_threshold": env.get_playbook_config["DEDUP_THRESHOLD"],
        "retrieval_threshold": env.get_playbook_config["RETRIEVAL_THRESHOLD"],
        "retrieval_topk": env.get_playbook_config['RETRIEVAL_TOP_K'],
    }

    logger.debug(f"provider : {request.llm_provider}")
    logger.debug(f"model : {request.llm_model}")

    async def event_generator():
        full_solution = ""
        # results of Retriever -> Generator
        captured_data = {}

        async for token in solution_stream(serving_graph, initial_state, captured_data):
            # SSE format
            payload = json.dumps({"token" : token}, ensure_ascii=False)
            yield f"data: {payload}\n\n"

            full_solution += token

        result_state = initial_state.copy()
        result_state.update(captured_data)
        result_state['solution'] = full_solution

        route = result_state.get("router_decision", "complex")

        if route == 'complex':
            asyncio.create_task(run_background_learning(result_state))

        # openAI format
        yield "data : [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type='text/event-stream')

if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=False)