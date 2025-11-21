import os
import asyncio
import csv

from datasets import load_dataset
from config.getenv import GetEnv
from graph.inference_graph import create_inference_graph
from utils import Logger
from module.db_management import get_db_instance

env = GetEnv()
logger = Logger(__name__)
save_logger = Logger(f"{__name__}_save", save_to_file=True, log_dir=env.get_log_dir, log_file="hotpotqa_config.log", console_output=False)

def format_hotpot_context(context_list):
    formatted_text = ""
    titles = context_list['title']
    sentences = context_list['sentences']
    
    for title, sent_list in zip(titles, sentences):
        formatted_text += f"Title: {title}\n"
        formatted_text += "".join(sent_list) + "\n\n"
    return formatted_text

async def main(num_sample : int = 200):
    db = get_db_instance()

    repo_id = "hotpotqa/hotpot_qa"
    dataset = load_dataset(repo_id, "distractor",  split="train")
    samples = dataset.select(range(num_sample))

    inference_graph = create_inference_graph()

    state = {
        "playbook": [], 
        "retrieved_bullets": [],
        
        # Config
        "max_playbook_size": env.get_eval_config["MAX_PLAYBOOK_SIZE"],
        "dedup_threshold": env.get_eval_config["DEDUP_THRESHOLD"],
        "retrieval_threshold": env.get_eval_config["RETRIEVAL_THRESHOLD"],
        "retrieval_topk" : env.get_eval_config['RETRIEVAL_TOP_K'],
        
        # HotpotQA용
        "ground_truth" : ""
    }

    csv_path = os.path.join(env.get_log_dir, 'hotpotqa_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # entry_point 대신 id 사용
        writer.writerow(["id", "is_success", "playbook_size", "retrieved_count", "helpful_count_in_retrieved"])

    save_logger.debug(f"max_playbook_size : {env.get_eval_config["MAX_PLAYBOOK_SIZE"]}")
    save_logger.debug(f"dedup_threshold : {env.get_eval_config["DEDUP_THRESHOLD"]}")
    save_logger.debug(f"retrieval_threshold : {env.get_eval_config["RETRIEVAL_THRESHOLD"]}")
    save_logger.debug(f"retrieval_topk : {env.get_eval_config["RETRIEVAL_TOP_K"]}")
    save_logger.debug(f"dataset sample : {num_sample}")

    
    for i, item in enumerate(samples):
        task_id = item['id']
        question = item['question']
        answer = item['answer']
        context_text = format_hotpot_context(item['context'])
        query = f"""Answer the question based on the context below.

[Context]
{context_text}

[Question]
{question}"""
        
        # State 주입
        state['query'] = query
        state['ground_truth'] = answer
        
        # 다른 태스크 필드는 비워줌 (충돌 방지)
        state['test_code'] = ""
        state['entry_point'] = ""

        # 실행 상태 초기화
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

        # CSV 저장
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([task_id, is_success, total_playbook_size, retrieved_count, helpful_hits])
            
        logger.info(f"Task: {task_id} | Success: {is_success} | DB: {total_playbook_size} | Retrieved: {retrieved_count}")

if __name__ == "__main__":
    asyncio.run(main())