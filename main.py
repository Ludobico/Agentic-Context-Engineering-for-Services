import os
import asyncio
from module.embed import EmbeddingPreprocessor
from module.database import PlayBookVectorStore
from module.graph import build_inference_graph, build_learning_graph
from module.node.ACE import evaluator_node
from config.getenv import GetEnv


env = GetEnv()
inference_graph = build_inference_graph()
learning_graph = build_inference_graph()
embedding_model = EmbeddingPreprocessor.default_embedding_model()
playbook_vector_store = PlayBookVectorStore()
vector_store = playbook_vector_store.get_or_create_store(embedding_model)

async def main():
    # 벡터스토어에서 메모리 로드
    playbook_memory = playbook_vector_store.load_all_playbook_entries(vector_store)

    if playbook_memory:
        current_step = max(entry['last_used_step'] for entry in playbook_memory) + 1
    else:
        current_step = 1

    while True:
        query = input("\n────────────────────────────────\nAsk a question (or 'exit' to quit):")

        if query.lower() == 'exit':
            break

        if not query.strip():
            continue

        inference_input = {
            "query" : query,
            "playbook" : [],
            "solution" : "",
            "verbose" : True,
            "retrieved_bullets" : [],
            "used_bullet_ids" : [],
            "trajectory" : [],
            "reflection" : {},
            "new_insights" : [],
            "feedback" : {},
            "current_step" : current_step,
            "max_playbook_size" : env.get_playbook_config['MAX_PLAYBOOK_SIZE'],
            "dedup_threshold" : env.get_playbook_config['DEDUP_THRESHOLD'],
            "retrieval_threshold" : env.get_playbook_config['RETRIEVAL_THRESHOLD']
        }

        config = {"configurable" : {"vector_store" : vector_store}}

        inference_result = await inference_graph.ainvoke(inference_input, config=config)

        evaluation_result = await evaluator_node(inference_result)

        learning_input = {
            **inference_result,
            **evaluation_result,
            "playbook" : playbook_memory,
            "current_step" : current_step,
        }

        learning_config = {
            "configurable" : {
                "playbook_vector_store" : playbook_vector_store,
                "vector_store" : vector_store,
                "embedding_model" : embedding_model
            }
        }
        learning_result = await learning_graph.ainvoke(learning_input, config=learning_config)

        playbook_memory = learning_result['playbook']
        current_step = learning_result['current_step']
    
if __name__ == "__main__":
    asyncio.run(main())
