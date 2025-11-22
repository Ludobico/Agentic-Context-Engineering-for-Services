import os
import asyncio
import csv

from datasets import load_dataset
from config.getenv import GetEnv
from graph.full_graph import create_full_graph
from utils import Logger
from module.db_management import get_db_instance

env = GetEnv()
logger = Logger(__name__)
save_logger = Logger(f"{__name__}_save", save_to_file=True, log_dir=env.get_log_dir, log_file="human_eval_config.log", console_output=False)

async def main():
    db = get_db_instance()

    repo_id = "openai/openai_humaneval"
    dataset = load_dataset(repo_id, split='test')

    # test_samples = dataset.select(range(20))
    test_samples = dataset

    inference_graph = create_full_graph()

    state = {
        "playbook": [], 
        "retrieved_bullets": [],
        
        # Config
        "max_playbook_size": env.get_playbook_config["MAX_PLAYBOOK_SIZE"],
        "dedup_threshold": env.get_playbook_config["DEDUP_THRESHOLD"],
        "retrieval_threshold": env.get_playbook_config["RETRIEVAL_THRESHOLD"],
        "retrieval_topk" : env.get_playbook_config['RETRIEVAL_TOP_K'],
        
        # HumanEval용 필드 (TypedDict에 추가 필요할 수 있음, 없으면 무시되거나 dict로 동작)
        "test_code": "",
        "entry_point": ""
    }

    csv_path = os.path.join(env.get_log_dir, 'human_eval_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["entry_point", "is_success", "playbook_size", "retrieved_count", "helpful_count_in_retrieved"])

    save_logger.debug(f"max_playbook_size : {env.get_playbook_config["MAX_PLAYBOOK_SIZE"]}")
    save_logger.debug(f"dedup_threshold : {env.get_playbook_config["DEDUP_THRESHOLD"]}")
    save_logger.debug(f"retrieval_threshold : {env.get_playbook_config["RETRIEVAL_THRESHOLD"]}")
    save_logger.debug(f"retrieval_topk : {env.get_playbook_config["RETRIEVAL_TOP_K"]}")

    for i, item in enumerate(test_samples):
        entry_point = item['entry_point']
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
        total_playbook_size = len(db.get_all_entries())
        retrieved_count = len(result.get('retrieved_bullets', []))
        helpful_hits = sum(1 for tag in result.get('reflection', {}).get('bullet_tags', []) if tag['tag'] == 'helpful')

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([entry_point, is_success, total_playbook_size, retrieved_count, helpful_hits])
            logger.info(f"task_id: {entry_point}, is_success: {is_success}, playbook_size: {total_playbook_size}, retrieved_count: {retrieved_count}, helpful_hits: {helpful_hits}")

if __name__ == "__main__":
    asyncio.run(main())

