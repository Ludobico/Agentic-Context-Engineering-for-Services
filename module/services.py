from module.graph import inference_graph, learning_graph
from module.node.ACE import evaluator_node
from module.state import State
from module.task import run_learning_task
from config.getenv import GetEnv

env = GetEnv()

class InferenceService:
    @staticmethod
    async def process_query(query : str) -> dict:
        state = State(query=query,
                      playbook=[],
                      solution=None,
                      retrieved_bullets=[],
                      used_bullet_ids=[],
                      trajectory=[],
                      reflection=None,
                      new_insights=None,
                      feedback=None,
                      current_step=0,
                      max_playbook_size=env.get_playbook_config['MAX_PLAYBOOK_SIZE'],
                      dedup_threshold=env.get_playbook_config['DEDUP_THRESHOLD'],
                      retrieval_threshhold=env.get_playbook_config['RETRIEVAL_THRESHOLD'],
                      verbose=True)
        inference_result = await inference_graph.ainvoke(state)
        evaluation_result = await evaluator_node(inference_result)

        payload = {**inference_result, **evaluation_result, "playbook" : inference_result['retrieved_bullets']}

        run_learning_task.delay(payload)

        print("🧠 학습 작업 백그라운드 전달 완료.")

        return {"solution": inference_result.get("solution", "답변을 생성하지 못했습니다.")}
    
class LearningService:
    @staticmethod
    async def update_playbook_from_disk(payload : dict):
        state = State(**payload, current_step=0)
        await learning_graph.ainvoke(state)
