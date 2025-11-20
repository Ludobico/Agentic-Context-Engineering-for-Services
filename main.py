import os
import asyncio
import shutil

from datasets import load_dataset
from config.getenv import GetEnv
from graph.inference_graph import create_inference_graph

env = GetEnv()

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

    for i, item in enumerate(test_samples):
        task_id = item['task_id']
        raw_prompt = item['prompt']
        query = f"complete the follwing task \n\n {raw_prompt}"

        test_code = item['test']
        entry_point = item['entry_point']

        state['query'] = query
        state['test_code'] = test_code
        state['entry_point'] = entry_point

        # Task별 초기화가 필요한 필드들 리셋
        state["solution"] = ""
        state["retrieved_bullets"] = []
        state["used_bullet_ids"] = []
        state["trajectory"] = []
        state["reflection"] = {}
        state["new_insights"] = []
        state["feedback"] = {}
        state['verbose'] = True

        result = await inference_graph.ainvoke(state)

        is_success = result.get("feedback", {}).get("rating") == "positive" 
        status_icon = "✅" if is_success else "❌"
        print(f"🚀 TASK {i+1}/10: {status_icon} {query[:80]}...")

if __name__ == "__main__":
    asyncio.run(main())

