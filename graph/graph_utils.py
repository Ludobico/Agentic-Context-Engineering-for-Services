from langgraph.graph.state import CompiledStateGraph
from typing import AsyncGenerator, Any
import re
from langgraph.graph.state import CompiledStateGraph
from PIL import Image
import os
import io
import json

from utils import highlight_print, Logger
from config.getenv import GetEnv

env = GetEnv()

logger = Logger(__name__)

async def solution_stream(graph : "CompiledStateGraph", input_data, capture_container: dict[str, Any] = None) -> AsyncGenerator:
    buffer = ""           
    solution_buffer = ""  
    
    state = "DETECTING" 
    is_escaped = False

    # 로그 추적 대상 노드
    NODE_LOG_MAP = {
        "router": "Router: Analyzing query...",
        "retriever": "Retriever: Searching Playbook...",
        "simple_generator": "Simple Generator: Generating response...",
        "generator": "Generator: Thinking with Playbook...",
        "evaluator": "Evaluator: Assessing response quality...",
        "reflector": "Reflector: Analyzing root causes & insights...",
        "curator": "Curator: Refining Playbook entries...",
        "update": "Update: Saving new knowledge to Database..."
    }

    async for event in graph.astream_events(input_data, version="v2"):
        
        # 1. 노드 시작 로그 (Log Streaming)
        if event["event"] == "on_chain_start":
            node_name = event.get("name")
            if node_name in NODE_LOG_MAP:
                yield {"type": "log", "content": NODE_LOG_MAP[node_name]}

        # 2. 데이터 캡처 (Data Capture)
        if event["event"] == "on_chain_end":
            data = event.get("data", {})
            output = data.get("output")
            if isinstance(output, dict) and capture_container is not None:
                # Router
                if "router_decision" in output:
                    capture_container["router_decision"] = output["router_decision"]
                # Retriever
                if "retrieved_bullets" in output:
                    capture_container["retrieved_bullets"] = output["retrieved_bullets"]
                    if "playbook" in output:
                        capture_container["playbook"] = output["playbook"]
                # Generator
                if "used_bullet_ids" in output:
                    capture_container["used_bullet_ids"] = output["used_bullet_ids"]
                if "trajectory" in output:
                    capture_container["trajectory"] = output["trajectory"]
                    if isinstance(output["trajectory"], list) and output["trajectory"]:
                        traj_text = output["trajectory"][0]
                        match = re.search(r'## Rationale \(Thought Process\):\n(.*?)\n\n## Solution', traj_text, re.DOTALL)
                        if match:
                            capture_container['rationale'] = match.group(1).strip()

        # 3. 텍스트 스트리밍 (Token Streaming)
        if event['event'] == "on_chat_model_stream":
            chunk = event['data']['chunk'].content
            if not chunk: continue

            # Simple Mode 감지
            is_simple_mode = False
            if capture_container and capture_container.get("router_decision") == "simple":
                is_simple_mode = True

            # [Case A] Simple Mode (Raw Text)
            if is_simple_mode:
                # [수정 2] buffer 대신 solution_buffer에 바로 누적
                solution_buffer += chunk
                yield {"type": "token", "content": chunk}
                continue

            # [Case B] Complex Mode (JSON Parsing)
            if state == "DETECTING":
                buffer += chunk # 임시 버퍼에 누적
                
                # "solution": " 패턴 찾기
                match = re.search(r'"solution"\s*:\s*"', buffer)
                
                if match:
                    state = "STREAMING_JSON"
                    # 매칭된 부분 뒷부분부터가 진짜 내용
                    remaining = buffer[match.end():]
                    buffer = "" # 임시 버퍼 리셋
                    
                    # 남은 뒷부분 처리 (이스케이프 로직 적용)
                    for char in remaining:
                        # 아래 STREAMING_JSON 로직과 동일하게 처리
                        if is_escaped:
                            char_yield = ""
                            if char == 'n': char_yield = '\n'
                            elif char == 't': char_yield = '\t'
                            elif char in ['"', '\\', '/']: char_yield = char
                            else: char_yield = f'\\{char}'
                            
                            solution_buffer += char_yield
                            yield {"type": "token", "content": char_yield}
                            is_escaped = False
                        elif char == '\\':
                            is_escaped = True
                        elif char == '"':
                            state = "DONE"
                            break
                        else:
                            solution_buffer += char
                            yield {"type": "token", "content": char}

                # 타임아웃 (너무 길어지면 그냥 출력)
                elif len(buffer) > 1000:
                    yield {"type": "token", "content": buffer}
                    solution_buffer += buffer
                    buffer = ""
            
            elif state == "STREAMING_JSON":
                for char in chunk:
                    if is_escaped:
                        char_yield = ""
                        if char == 'n': char_yield = '\n'
                        elif char == 't': char_yield = '\t'
                        elif char in ['"', '\\', '/']: char_yield = char
                        else: char_yield = f'\\{char}'
                        
                        solution_buffer += char_yield
                        yield {"type": "token", "content": char_yield}
                        is_escaped = False
                    elif char == '\\':
                        is_escaped = True
                    elif char == '"':
                        state = "DONE"
                        break
                    else:
                        solution_buffer += char
                        yield {"type": "token", "content": char}

    # 최종 저장 (Learning Graph용)
    if capture_container is not None:
        # 만약 buffer에 남은 게 있다면 (타임아웃 등) 붙여주기
        final_solution = solution_buffer + buffer if state != "DONE" else solution_buffer
        capture_container['solution'] = final_solution

def graph_to_png(compiled_graph : "CompiledStateGraph", show_direct : bool = True):
    """
    given a compiled langgraph, generate a graph image of that using mermaid,
    and either show it directly or save it to a file in `temp folder`.

    Parameters
    ----------
    compiled_graph : CompiledStateGraph
        the compiled langgraph to be visualized
    show_direct : bool, optional
        whether to show the image directly or save it to a file, by default True
    """
    graph_data = compiled_graph.get_graph().draw_mermaid_png()
    image = Image.open(io.BytesIO(graph_data))

    if show_direct:
        image.show()
    else:
        output_path = os.path.join(env.get_log_dir, 'graph.png')
        image.save(output_path, format='png')
        highlight_print(f"graph image is saved at {output_path}", 'green')

def initialize_langsmith_tracking(show_tracking : bool = True):
    env = GetEnv()
    """
    Tracking log via langsmith\n
    Set `show_tracking` to False if you don't want to display in the terminal
    """
    if not env.get_monitoring_enabled:
        return
    
    langsmith_api_key = env.get_langsmith_api_key

    project_name = env.get_langsmith_project_name

    if not langsmith_api_key or not project_name:
        if show_tracking:
            logger.warning("LangSmith Monitor is True, but API Key or Project Name is missing")
            return

    # activate V2 tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "true" 
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = project_name

    # cyan
    if show_tracking:
        highlight_print(project_name, color='cyan')