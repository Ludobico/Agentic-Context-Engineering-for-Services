from langgraph.graph.state import CompiledStateGraph
from typing import AsyncGenerator
import re

async def solution_stream(graph : "CompiledStateGraph", input_data) -> AsyncGenerator:
    buffer = ""
    
    # SEARCHING,STREAMING,DONE
    state = "SEARCHING"
    is_escaped = False

    async for event in graph.astream_events(input_data, version="v2"):
        # if state == "DONE":
        #     break

        if event['event'] == "on_chat_model_stream":
            chunk = event['data']['chunk'].content
            if not chunk:
                continue

            if state == "SEARCHING":
                buffer += chunk
                match = re.search(r'"solution"\s*:\s*"', buffer)

                if match:
                    state = "STREAMING"

                    remaining_chars = buffer[match.end():]

                    chunk = remaining_chars
                else:
                    continue

            if state == "STREAMING":
                for char in chunk:
                    if is_escaped:
                        if char == 'n':
                            yield '\n'
                        elif char == 't':
                            yield '\t'
                        elif char == '"' or char == '\\' or char == '/':
                            yield char
                        else:
                            yield f'\\{char}'
                        is_escaped = False
                    elif char == '\\':
                        is_escaped = True
                    elif char == '"':
                        state = "DONE"
                        break
                    else:
                        yield char
