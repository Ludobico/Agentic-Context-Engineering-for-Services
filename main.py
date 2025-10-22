import asyncio
import time
import os

from module.services import InferenceService
from config.getenv import GetEnv

env = GetEnv()

async def run_test():
    test_query = "파이썬으로 간단한 웹 서버 만드는 코드를 알려줘"

    start_time = time.time()

    result = await InferenceService.process_query(test_query)
    
    duration = time.time() - start_time
    
    print("\n" + "-" * 30)
    print(f"✅ 테스트 성공! (소요 시간: {duration:.2f}초)")
    
    # 3. 결과 출력
    print("\n[최종 반환된 솔루션]:")
    print(result.get("solution", "솔루션이 없습니다."))
    
    print("\n---")
    print("ℹ️ 참고: 이 테스트는 '추론' 파이프라인만 실행합니다.")
    print("Celery 작업이 Redis에 전송되었지만, '학습' 파이프라인(Reflector 등)은")
    print("별도의 Celery 워커가 실행되어야 처리됩니다.")

if __name__ == "__main__":
    asyncio.run(run_test())