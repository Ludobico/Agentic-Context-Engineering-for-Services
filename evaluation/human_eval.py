import os
import asyncio
import csv

from datasets import load_dataset
from config.getenv import GetEnv
from graph.inference_graph import create_inference_graph
from utils import Logger

env = GetEnv()
logger = Logger(__name__)

async def main():
    repo_id = "openai/openai_humaneval"
    dataset = load_dataset(repo_id, split='test')

    test_samples = dataset.select(range(10))

    inference_graph = create_inference_graph()

    state = {
        "playbook": [], 
        "retrieved_bullets": [],
        
        # Config
        "max_playbook_size": env.get_playbook_config["MAX_PLAYBOOK_SIZE"],
        "dedup_threshold": env.get_playbook_config["DEDUP_THRESHOLD"],
        "retrieval_threshold": env.get_playbook_config["RETRIEVAL_THRESHOLD"],
        
        # HumanEval용 필드 (TypedDict에 추가 필요할 수 있음, 없으면 무시되거나 dict로 동작)
        "test_code": "",
        "entry_point": ""
    }

    csv_path = os.path.join(env.get_log_dir, 'human_eval_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "is_success", "playbook_size", "retrieved_count", "helpful_count_in_retrieved"])

    for i, item in enumerate(test_samples):
        task_id = item['task_id']
        raw_prompt = item['prompt']
        query = f"complete the follwing task \n\n {raw_prompt}"

        test_code = item['test']
        entry_point = item['entry_point']

        state['query'] = query
        state['test_code'] = test_code
        state['entry_point'] = entry_point

        state["solution"] = ""
        state["retrieved_bullets"] = []
        state["used_bullet_ids"] = []
        state["trajectory"] = []
        state["reflection"] = {}
        state["new_insights"] = []
        state["feedback"] = {}
        state['verbose'] = True

        result = await inference_graph.ainvoke(state)

        is_success = 1 if result['feedback']['rating'] == 'positive' else 0
        playbook_size = len(state['playbook'])
        retrieved_count = len(state.get('retrieved_bullets', []))
        helpful_hits = sum(1 for tag in state.get('reflection', {}).get('bullet_tags', []) if tag['tag'] == 'helpful')

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([task_id, is_success, playbook_size, retrieved_count, helpful_hits])
            logger.info(f"task_id: {task_id}, is_success: {is_success}, playbook_size: {playbook_size}, retrieved_count: {retrieved_count}, helpful_hits: {helpful_hits}")

if __name__ == "__main__":
    asyncio.run(main())

