from celery import Celery
from config.getenv import GetEnv

env = GetEnv()

celery = Celery(__name__, broker=env.get_celery_broker_url, backend=env.get_celery_result_backend, include=['app.tasks'])
celery.conf.update(task_serializer="json", accept_content=['json'], result_serializer='json', timezone='Asia/Seoul', enable_utc=True)

