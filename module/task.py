import asyncio
from module.celery_app import celery
from module.services import LearningService

@celery.task(name="run_larning_task")
def run_learning_task(payload: dict):
    print(f" 학습 작업 수신: {payload['query'][:50]}...")
    asyncio.run(LearningService.update_playbook_from_disk(payload))