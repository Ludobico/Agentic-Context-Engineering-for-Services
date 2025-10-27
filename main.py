import os
import asyncio
from module.embed import EmbeddingPreprocessor
from module.database import PlayBookVectorStore
from module.graph import build_inference_graph, build_learning_graph
from module.node.ACE import evaluator_node
from config.getenv import GetEnv


env = GetEnv()
inference_graph = build_inference_graph()
learning_graph = build_learning_graph()
embedding_model = EmbeddingPreprocessor.default_embedding_model()
playbook_vector_store = PlayBookVectorStore()
vector_store = playbook_vector_store.get_or_create_store(embedding_model)

tasks = [
        {
            "query": """
다음 요구사항을 만족하는 파이썬 함수 `analyze_sales`를 작성해줘:
- 입력: 여러 지점의 판매 데이터를 담은 딕셔너리 리스트
  예: [{"store": "A", "sales": [100, 200, None, 150]}, {"store": "B", "sales": [300, 400]}]
- 출력: 각 지점의 평균 판매액을 딕셔너리로 반환 (None 값은 제외)
- 예외 처리: sales가 비어있거나, 모든 값이 None이면 평균은 0으로 처리
"""
        },
        {
            "query": """
이전에 만든 `analyze_sales` 함수를 확장해서, 외부 API에서 판매 데이터를 가져오는 시뮬레이션을 추가해줘.
- `fetch_sales_data(store_id)` 함수를 만들어서, time.sleep(2)로 네트워크 지연을 시뮬레이션
- 3개 지점 (A, B, C)의 데이터를 순차적으로 가져와서 분석
- 전체 실행 시간을 측정해서 출력
(힌트: 비동기 처리를 사용하지 않으면 매우 느릴 것)
"""
        },
        {
            "query": """
이전 단계의 코드가 너무 느려! (6초 이상 소요됨)
asyncio 또는 concurrent.futures를 사용해서 병렬로 데이터를 가져오도록 개선해줘.
- 목표: 전체 실행 시간을 3초 이하로 단축
- 각 지점의 데이터를 동시에 가져와서 처리
- 결과는 이전과 동일하게 각 지점의 평균 판매액 딕셔너리로 반환
"""
        },
        {
            "query": """
이제 1000개 지점의 데이터를 처리해야 해. 각 지점은 10,000개의 판매 기록을 가지고 있어.
현재 코드를 그대로 사용하면 메모리가 부족할 수 있어.
- Generator를 사용해서 메모리 효율적으로 데이터를 처리하도록 개선
- 전체 데이터를 한 번에 메모리에 올리지 말고, 스트리밍 방식으로 처리
- 상위 10개 지점의 평균 판매액만 반환 (heapq 사용 권장)
"""
        },
        {
            "query": """
실제 운영 환경에서는 API 호출이 간헐적으로 실패할 수 있어.
- fetch_sales_data 함수를 수정해서, 30% 확률로 random하게 ConnectionError를 발생시켜
- 실패 시 최대 3번까지 재시도하는 로직 추가 (exponential backoff 적용: 1초, 2초, 4초)
- 3번 모두 실패하면 해당 지점 데이터는 건너뛰고, 로그 메시지 출력
- 전체 1000개 지점 중 성공한 지점만으로 상위 10개 계산
"""
        },
        {
            "query": """
마지막으로 전체 시스템을 검증하는 테스트 코드를 작성해줘:
1. 빈 데이터셋 테스트 (빈 리스트)
2. 모든 API 호출이 실패하는 경우 테스트
3. 일부 지점의 sales가 [None, None, None]인 경우
4. 음수 판매액이 포함된 경우 (validation 추가 필요)
5. 멀티스레딩 환경에서 race condition이 발생하지 않는지 검증

각 테스트는 assert 문으로 검증하고, 통과/실패 여부를 명확히 출력해줘.
"""}]

async def main():
    # 벡터스토어에서 메모리 로드
    playbook_memory = playbook_vector_store.load_all_playbook_entries(vector_store)

    if playbook_memory:
        current_step = max(entry['last_used_step'] for entry in playbook_memory) + 1
    else:
        current_step = 1

    # while True:
    #     query = input("\n────────────────────────────────\nAsk a question (or 'exit' to quit):")

    #     if query.lower() == 'exit':
    #         break

    #     if not query.strip():
    #         continue
    for task in tasks:
        query = task['query']

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
