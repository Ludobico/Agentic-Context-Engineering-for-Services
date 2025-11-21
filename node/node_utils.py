import tiktoken
import re
import json
from datetime import datetime
from typing import Any, Optional
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from module.db_management import VectorStore
from core.state import PlaybookEntry
from config.getenv import GetEnv

env = GetEnv()

def token_calculator(text : str) -> int:
    encoding = tiktoken.get_encoding("o200k_base")
    tokens = encoding.encode(text)
    return len(tokens)

class SolutionOnlyStreamCallback(AsyncCallbackHandler):
    def __init__(self):
        self.buffer = ""
        self.in_solution = False
        self.solution_content = ""

    async def on_llm_new_token(self, token : str, **kwargs):
        self.buffer += token
    
        # solution pattern
        if '"solution"' in self.buffer and not self.in_solution:
            match = re.search(r'"solution"\s*:\s*"', self.buffer)
            if match:
                self.in_solution = True
        
        if self.in_solution:
            if token == '"' and self.buffer.count('"solution"') > 0:
                closing_quote_count = self.buffer[self.buffer.rfind('"solution"'):].count('"')
                if closing_quote_count >= 3:
                    self.in_solution = False
                    return
        
            clean_token = token.replace('\\n', '\n').replace('\\"', '"')
            print(clean_token, end="", flush=True)
            self.solution_content += clean_token

class StrictJsonOutputParser(JsonOutputParser):
    def parse(self, text: str) -> dict[str, Any]:
        cleaned_text = self._preprocess(text)
        
        strategies = [
            self._parse_direct,
            self._parse_code_block,
            self._parse_first_json,
            self._parse_between_braces,
            self._parse_fix_common_errors
        ]
        
        last_error = None
        for strategy in strategies:
            try:
                result = strategy(cleaned_text)
                if result is not None:  # 명시적으로 None 체크
                    return result
            except Exception as e:
                last_error = e
                continue
        
        # 모든 전략 실패 시 명시적 에러 발생
        raise ValueError(
            f"Failed to parse JSON from text. Last error: {last_error}\n"
            f"Text preview: {cleaned_text[:200]}..."
        )


    def _preprocess(self, text : str) -> str:
        text = text.strip()
        text = text.replace('\ufeff', '')
        return text
    
    def _parse_direct(self, text : str) -> dict[str, Any]:
        return json.loads(text)
    
    def _parse_code_block(self, text : str) -> dict[str, Any]:
        patterns = [
            r'```json\s*\n(.*?)\n```',  # ```json
            r'```\s*\n(.*?)\n```',      # ```
            r'`(.*?)`',                 
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                try:
                    return json.loads(json_str)
                except:
                    continue
        return None
    
    def _parse_first_json(self, text : str) -> dict[str, Any]:
        start_idx = text.find('{')
        if start_idx == -1:
            return None
        
        brace_count = 0
        in_string = False
        escape = False

        for i in range(start_idx, len(text)):
            char = text[i]

            if char == '"' and not escape:
                in_string = not in_string

            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1

                    if brace_count == 0:
                        # json
                        json_str = text[start_idx:i+1]
                        try:
                            return json.loads(json_str)
                        except:
                            return None
            escape = (char == '\\' and not escape)

        return None
    
    def _parse_between_braces(self, text : str) -> dict[str, Any]:
        first_brace = text.find('{')
        last_brace = text.find('}')

        if first_brace == -1 or last_brace == -1:
            return None
        
        json_str = text[first_brace:last_brace + 1]

        try:
            return json.loads(json_str)
        except:
            return None
        
    def _parse_fix_common_errors(self, text : str) -> dict[str, Any]:
        text = re.sub(r'```json\s*\n?', '', text)
        text = re.sub(r'```\s*\n?', '', text)
        text = re.sub(r'`', '', text)
        text = re.sub(r'\n\s*', ' ', text)
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        text = re.sub(r"'([^']*)'(\s*:)", r'"\1"\2', text)
        text = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)

        try:
            return json.loads(text)
        except:
            return None

def prune_playbook(playbook : PlaybookEntry, max_size : int) -> tuple[list[PlaybookEntry], list[str]]:
    """
    Prunes the playbook based on two criteria:
    1. Quality: Remove entries where harmful_count > helpful_count (Poisoned knowledge)
    2. Capacity: If size > max_size, remove entries with lowest value (LRU & Low Utility)
    """
    ids_to_prune = set()
    kept_entries = []

    clean_entries = []
    for entry in playbook:
        h_count = entry.get("helpful_count", 0)
        harm_count = entry.get("harmful_count", 0)

        if harm_count > h_count:
            ids_to_prune.add(entry['entry_id'])
        else:
            clean_entries.append(entry)
    
    current_size = len(clean_entries)

    if current_size > max_size:
        excess_count = current_size - max_size

        clean_entries.sort(
            key=lambda x : (
                x.get("helpful_count", 0),
                x.get("last_used_at") or datetime.min,
            )
        )

        entries_to_remove = clean_entries[:excess_count]
        kept_entries = clean_entries[excess_count:]

        for entry in entries_to_remove:
            ids_to_prune.add(entry['entry_id'])
    
    else:
        kept_entries = clean_entries

    return kept_entries, list(ids_to_prune)

def is_duplicate_entry(
        content : str,
        vector_store : VectorStore,
        embedding_model : Optional[HuggingFaceEmbeddings] = None,
        threshold : Optional[float] = None
    ) -> bool:
    if threshold is None:
        threshold = float(env.get_eval_config['DEDUP_THRESHOLD'])
    
    if embedding_model is None:
        embedding_model = VectorStore.get_embedding_model

        if vector_store.get_doc_count() == 0:
            return False
        
        retriever = vector_store.from_disk()
        query_embedding = embedding_model.embed_query(content)

        similar_docs = retriever.similarity_search_by_vector(
            embedding=query_embedding,
            k=1,
            score_threshold=threshold
        )

        return bool(similar_docs)

def run_human_eval_test(
        generated_code : str,
        test_code : str,
        test_id : str
):
    full_code = f"{generated_code}\n\n{test_code}\n\ncheck({test_id})"

    try:
        exec_globals = {}
        exec(full_code, exec_globals)
        return True, "All unit tests passed successfully."
    except AssertionError:
        return False, "Unit tests failed."
    except Exception as e:
        return False, f"An error occurred: {e}"
    
def run_hotpot_eval_test(
        generated_answer : str,
        ground_truth : str
) -> tuple[bool, str]:
    if not generated_answer:
        return False, "Generated answer is empty."
    
    gen_norm = str(generated_answer).lower().strip()
    truth_norm = str(ground_truth).lower().strip()

    if gen_norm == truth_norm:
        return True, f"Exact Match! Answer '{ground_truth}' found."
    
    if truth_norm in gen_norm:
        return True, f"Partial Match! Answer '{ground_truth}' found in response."
        
    return False, f"Mismatch. Expected '{ground_truth}', but got '{generated_answer}'."