import os
import asyncio
import shutil

from config.getenv import GetEnv
from graph import create_learning_graph, create_serving_graph
from graph.graph_utils import solution_stream
from core.state import State
from utils import highlight_print

env = GetEnv()

inference_graph = create_serving_graph()
serving_graph = create_learning_graph()

TEST_QUERIES = [
    "파이썬에서 리스트를 역순으로 정렬하는 방법은?",
    "자바스크립트에서 비동기 처리를 하는 async/await 예제 보여줘",
    "SQL에서 중복된 행을 제거하는 쿼리는?",
    "리액트 useEffect 훅의 사용법과 주의사항 알려줘"
]

async def run_ace_pipeline(state: State, task_id: int):
    solution = ""
    captured_data = {}

    print(f"\n💬 [To User-{task_id}]: ", end="")
    
    # [수정 1] inference_graph 대신 serving_graph 사용 (여기선 Generator까지만 실행)
    async for token in solution_stream(inference_graph, state, captured_data):
        print(token, end="", flush=True)
        solution += token
    
    print("\n") # 줄바꿈

    # State 업데이트
    state.update(captured_data)
    state['solution'] = solution

    # [수정 2] 백그라운드 학습 시작 (Non-blocking)
    print(f"📚 [System-{task_id}] Background Learning Started...")
    asyncio.create_task(run_background_learning(state, task_id))
    
    return solution

async def run_background_learning(state_from_inference: State, task_id: int):
    try:
        # 실제 학습 그래프 실행 (Evaluator -> ... -> Update)
        await serving_graph.ainvoke(state_from_inference)
        print(f"✅ [System-{task_id}] Learning Completed & DB Updated!")
    except Exception as e:
        print(f"⚠️ [System-{task_id}] Learning Failed: {e}")

async def main():
    # 공통 State 설정
    base_config = {
        "playbook": [], 
        "retrieved_bullets": [],
        "max_playbook_size": env.get_playbook_config["MAX_PLAYBOOK_SIZE"],
        "dedup_threshold": env.get_playbook_config["DEDUP_THRESHOLD"],
        "retrieval_threshold": env.get_playbook_config["RETRIEVAL_THRESHOLD"],
        "retrieval_topk": env.get_playbook_config['RETRIEVAL_TOP_K'],
        # 테스트용 빈 값들
        "test_code": "", "entry_point": "", "ground_truth": ""
    }

    print("=== 🚀 Async ACE Pipeline Test Started ===")

    # [수정 3] 여러 질문을 연속으로 던짐
    for i, query in enumerate(TEST_QUERIES):
        # 각 질문마다 새로운 State 생성
        state = base_config.copy()
        state['query'] = query
        state['verbose'] = False # 로그 너무 많으면 헷갈리니까 끔

        # 파이프라인 실행 (답변만 받고 즉시 리턴됨)
        await run_ace_pipeline(state, i+1)
        
        # 사용자가 다음 질문 하기 전 딜레이 (1초)
        # 이 사이에 백그라운드 로그가 끼어드는지 보세요!
        await asyncio.sleep(1)

    print("\n=== 🛑 All user queries finished. Waiting for background tasks... ===")
    
    # [수정 4] 메인 스레드 생존 유지 (이게 없으면 백그라운드 작업이 강제 종료됨)
    # 실제 서버에서는 필요 없지만, 테스트 스크립트에서는 필수!
    for _ in range(15):
        print(".", end="", flush=True)
        await asyncio.sleep(1)
    
    print("\n=== Test Finished ===")

if __name__ == "__main__":
    asyncio.run(main())