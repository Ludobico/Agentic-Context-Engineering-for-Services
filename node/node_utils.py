import tiktoken
import re
import json
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
    def parse(self, text : str) -> dict[str, Any]:
        cleand_text = self._preprocess(text)

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
                result = strategy(cleand_text)
                if result:
                    return result
            except Exception as e:
                last_error = e
                continue


    def _preprocess(self, text : str) -> str:
        text = text.strip()
        text = text.replace('\ufeff', '')
        return text
    
    def _parse_direct(self, text : str) -> dict[str, Any]:
        return json.loads()
    
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

def prune_playbook(playbook : PlaybookEntry) -> PlaybookEntry:
    """
    Prunes the playbook by removing entries where `harmful_count` > `helpful_count`
    """
    ids_to_prune = []

    for entry in playbook:
        if entry.get("harmful_count", 0) > entry.get("helpful_count", 0):
            ids_to_prune.append(entry['entry_id'])
        
    pruned_playbook = [entry for entry in playbook if entry['entry_id'] not in ids_to_prune]

    return pruned_playbook, ids_to_prune

def is_duplicate_entry(
        content : str,
        vector_store : VectorStore,
        embedding_model : Optional[HuggingFaceEmbeddings] = None,
        threshold : Optional[float] = None
    ) -> bool:
    if threshold is None:
        threshold = float(env.get_playbook_config['DEDUP_THRESHOLD'])
    
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
        entry_point : str
):
    full_code = f"{generated_code}\n\n{test_code}\n\ncheck({entry_point})"

    try:
        exec_globals = {}
        exec(full_code, exec_globals)
        return True, "All unit tests passed successfully."
    except AssertionError:
        return False, "Unit tests failed."
    except Exception as e:
        return False, f"An error occurred: {e}"