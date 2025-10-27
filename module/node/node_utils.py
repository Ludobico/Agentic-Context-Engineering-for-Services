from typing import List
from module.state import PlaybookEntry
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings

class DeltaJsonParser:
    def __init__(self):
        self.previous_values = {}
        self.final_values = {} 

    def parse_delta(self, chunk: dict):
        deltas = {}

        for key, value in chunk.items():
            self.final_values[key] = value
            
            if isinstance(value, str):
                prev = self.previous_values.get(key, "")
                if value.startswith(prev):
                    deltas[key] = value[len(prev):]
                    self.previous_values[key] = value
                else:
                    deltas[key] = value
                    self.previous_values[key] = value
            elif isinstance(value, list):
                deltas[key] = value
                
        return deltas
    
    def get_final_values(self):
        return self.final_values.copy()
    
    def reset(self):
        self.previous_values = {}
        self.final_values = {}

def deduplicate_playbook(playbook : List[PlaybookEntry], threshold : float, embedding_model : HuggingFaceEmbeddings) -> List[PlaybookEntry]:
    if len(playbook) <= 1:
        return playbook
    
    contents = [entry['content'] for entry in playbook]
    embeddings = embedding_model.embed_documents(contents)

    similarity_matrix = cosine_similarity(embeddings)

    to_remove = set()

    for i in range(len(playbook)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(playbook)):
            if j in to_remove:
                continue

            if similarity_matrix[i][j] >= threshold:
                if playbook[i]['helpful_count'] >= playbook[j]['helpful_count']:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
                    break
    
    return [entry for idx, entry in enumerate(playbook) if idx not in to_remove]


def prune_playbook(playbook : List[PlaybookEntry], max_size : int) -> List[PlaybookEntry]:
    playbook = [
        entry for entry in playbook
        if entry['helpful_count'] >= entry['harmful_count']
    ]

    if len(playbook) <= max_size:
        return playbook
    
    def score(entry):
        total_usage = entry['helpful_count'] + entry['harmful_count']
        if total_usage == 0:
            return entry['last_used_step']
        
        return (entry['helpful_count'] - entry['harmful_count']) + entry['last_used_step'] * 0.1
    
    playbook.sort(key=score, reverse=True)

    return playbook[:max_size]