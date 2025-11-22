import requests
import json
import time

from utils import Logger

logger = Logger(__name__)

url = "http://localhost:8000/chat/stream"
TEST_QUERIES = [
    "파이썬에서 리스트를 정렬하는 가장 기초적인 방법은?",
    "그럼 리스트를 역순으로 정렬하려면 어떻게 해?", 
    "딕셔너리의 키를 기준으로 정렬하는 방법도 알려줘",
    "정렬 속도가 가장 빠른 알고리즘은 뭐야?"
]

def question(query, turn):
    print(f"[USER TURN : {turn}]")
    print(f"Query : {query}")

    payload = {"query" : query}
    start_time = time.time()

    with requests.post(url, json=payload, stream=True) as r:
        for line in r.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data:"):
                    data_str = decoded_line[5: ].strip()
                    if data_str == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)
                        print(data['token'], end="", flush=True)

                    except:
                        pass
    
    end_time = time.time()
    duration = end_time - start_time

    logger.debug(f"\nDuration : {duration}")

def main():
    for i, query in enumerate(TEST_QUERIES):
        question(query, i+1)


if __name__ == "__main__":
    main()


