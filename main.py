import uvicorn
import os
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import asyncio
import json
from contextlib import asynccontextmanager
import torch

from utils import Logger
from core import State, ChatRequest
from graph import create_serving_graph, create_learning_graph, create_full_graph
from graph.graph_utils import solution_stream
from config.getenv import GetEnv
from module.memory import RedisMemoryManager
from module.db_management import get_db_instance, get_vector_store_instance, reset_all_stores

env = GetEnv()
logger = Logger(__name__)

serving_graph = None
learning_graph = None
full_graph = None
memory_manager = None
backend_port = int(os.getenv("BACKEND_PORT"))

@asynccontextmanager
async def lifespan(app : FastAPI):
    global serving_graph, learning_graph, full_graph, memory_manager
    torch.cuda.empty_cache()

    # graph
    serving_graph = create_serving_graph()
    learning_graph = create_learning_graph()
    full_graph = create_full_graph()
    # memory
    memory_manager = RedisMemoryManager()

    yield

app = FastAPI(title="ACE Framework API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def run_background_learning(state : State):
    await learning_graph.ainvoke(state)

@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id : str):
    history = await memory_manager.get_history(session_id)
    return history

@app.get("/chat/sessions")
async def get_sessions():
    sessions = await memory_manager.get_all_session_ids()
    return {"sessions" : sessions}

@app.delete("/chat/history/{session_id}")
async def delete_chat_history(session_id : str):
    await memory_manager.clear_session(session_id)
    return {"status" : "success"}

@app.post("/chat/stream")
async def chat_stream(request : ChatRequest):

    print(request.execution_mode)
    if request.execution_mode == 'full':
        target_graph = full_graph
    else:
        target_graph = serving_graph
    
    sid = request.session_id
    initial_state = {
        "query" : request.query,
        "playbook" : [],
        "solution" : "",
        "verbose" : False,
        "router_decision" : "",
        "session_id" : sid,
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

    # memory : question
    await memory_manager.save_user_message(sid, request.query)

    logger.debug(f"provider : {request.llm_provider}")
    logger.debug(f"model : {request.llm_model}")

    async def event_generator():
        full_solution = ""
        # results of Retriever -> Generator
        captured_data = {}

        async for token in solution_stream(target_graph, initial_state, captured_data):
            # SSE format
            payload = json.dumps(token, ensure_ascii=False)
            yield f"data: {payload}\n\n"

            if token['type'] == 'token':
                full_solution += token['content']
                

        result_state = initial_state.copy()
        result_state.update(captured_data)
        result_state['solution'] = full_solution

        # memory : answer
        result_state['session_id'] = sid
        await memory_manager.save_ai_message(sid, full_solution)

        if request.execution_mode == 'standard':
            route = result_state.get("router_decision", "complex")
            if route == 'complex':
                # serving과 learning은 분리되어있어서 solution_stream으로 learning graph의 로그를 보여줄 수 없음
                # log_msg = {"type" : "log", "content" : "Background Learning..."}
                # yield f"data : {json.dumps(log_msg, ensure_ascii=False)}"
                asyncio.create_task(run_background_learning(result_state))
            else:
                # log_msg = {"type" : "log", "content" : "Full Cycle Completed"}
                # yield f"data: {json.dumps(log_msg, ensure_ascii=False)}"
                pass

        # openAI format
        yield "data : [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type='text/event-stream')

@app.get("/playbook/stats")
async def get_playbook_stats():
    try:
        vs = get_vector_store_instance()
        count = vs.get_doc_count()
        return {"status" : "success", "count" : count}
    
    except Exception as e:
        return {"status" : "error", "count" : 0, "message" : str(e)}
    
@app.delete("/playbook/reset")
async def reset_playbook():
    reset_all_stores(target='both')
    return {"status" : "success", "message" : "Playbook reset complete"}

if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=int(backend_port), reload=False)