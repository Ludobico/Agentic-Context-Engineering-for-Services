from langgraph.graph.state import CompiledStateGraph
from typing import AsyncGenerator, Any
import re
from langgraph.graph.state import CompiledStateGraph
from PIL import Image
import os
import io

from utils import highlight_print
from config.getenv import GetEnv

env = GetEnv()

async def solution_stream(graph : "CompiledStateGraph", input_data, capture_container: dict[str, Any] = None) -> AsyncGenerator:
    buffer = ""
    # SEARCHING,STREAMING,DONE
    state = "SEARCHING"
    is_escaped = False

    async for event in graph.astream_events(input_data, version="v2"):
        if event["event"] == "on_chain_end":
            node_name = event.get("name")
            # Retriever나 Generator가 뱉은 결과(Dict)를 캡처
            if node_name in ["retriever", "generator"]:
                if capture_container is not None:
                    # 기존 컨테이너에 결과 병합 (retrieved_bullets, used_bullet_ids 등)
                    capture_container.update(event["data"]["output"])

        if event['event'] == "on_chat_model_stream":
            chunk = event['data']['chunk'].content
            if not chunk:
                continue

            if state == "SEARCHING":
                buffer += chunk
                # "solution": " 패턴 찾기
                match = re.search(r'"solution"\s*:\s*"', buffer)

                if match:
                    state = "STREAMING"
                    remaining_chars = buffer[match.end():]
                    chunk = remaining_chars
                else:
                    continue

            if state == "STREAMING":
                for char in chunk:
                    # 이스케이프 문자 처리 로직 (기존 유지)
                    if is_escaped:
                        if char == 'n': yield '\n'
                        elif char == 't': yield '\t'
                        elif char in ['"', '\\', '/']: yield char
                        else: yield f'\\{char}'
                        is_escaped = False
                    elif char == '\\':
                        is_escaped = True
                    elif char == '"':
                        state = "DONE"
                        break # for loop break
                    else:
                        yield char

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