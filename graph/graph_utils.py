from langgraph.graph.state import CompiledStateGraph
from typing import AsyncGenerator, Any
import re
from langgraph.graph.state import CompiledStateGraph
from PIL import Image
import os
import io
import json

from utils import highlight_print
from config.getenv import GetEnv

env = GetEnv()

async def solution_stream(graph: "CompiledStateGraph", input_data, capture_container: dict[str, Any] = None) -> AsyncGenerator:
    solution_buffer = ""
    state = "SEARCHING"
    is_escaped = False

    async for event in graph.astream_events(input_data, version='v2'):
        # ✅ Generator 완료 시 모든 메타데이터 캡처
        if event['event'] == 'on_chain_end' and event.get("name") == "generator":
            if capture_container is not None:
                output = event['data']['output']
                capture_container['used_bullet_ids'] = output.get('used_bullet_ids', [])
                capture_container['trajectory'] = output.get('trajectory', [])
                
                # Trajectory에서 rationale 추출
                if output.get('trajectory'):
                    trajectory_text = output['trajectory'][0]
                    match = re.search(r'## Rationale \(Thought Process\):\n(.*?)\n\n## Solution', 
                                    trajectory_text, re.DOTALL)
                    if match:
                        capture_container['rationale'] = match.group(1).strip()
        
        # Retriever 결과 캡처
        if event['event'] == 'on_chain_end' and event.get("name") == "retriever":
            if capture_container is not None:
                output = event['data']['output']
                if 'retrieved_bullets' in output:
                    capture_container['retrieved_bullets'] = output['retrieved_bullets']
        
        # Solution만 스트리밍
        if event['event'] == "on_chat_model_stream":
            chunk = event['data']['chunk'].content
            if not chunk:
                continue

            # SEARCHING 상태에서 "solution": " 찾기
            if state == "SEARCHING":
                solution_buffer += chunk
                match = re.search(r'"solution"\s*:\s*"', solution_buffer)
                if match:
                    state = "STREAMING"
                    solution_buffer = ""  # 리셋
                    remaining = chunk[match.end() - len(chunk):]
                    if remaining:
                        chunk = remaining
                    else:
                        continue
                else:
                    continue
                    
            if state == "STREAMING":
                for char in chunk:
                    if is_escaped:
                        if char == 'n': 
                            solution_buffer += '\n'
                            yield '\n'
                        elif char == 't': 
                            solution_buffer += '\t'
                            yield '\t'
                        elif char in ['"', '\\', '/']: 
                            solution_buffer += char
                            yield char
                        else: 
                            solution_buffer += f'\\{char}'
                            yield f'\\{char}'
                        is_escaped = False
                    elif char == '\\':
                        is_escaped = True
                    elif char == '"':
                        state = "DONE"
                        break
                    else:
                        solution_buffer += char
                        yield char
    
    # 최종 solution 저장
    if capture_container is not None:
        capture_container['solution'] = solution_buffer

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