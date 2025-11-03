from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from module.prompt import curator_prompt, evaluator_prompt, generator_prompt, reflector_prompt
from node.node_utils import SolutionOnlyStreamCallback, StrictJsonOutputParser
from core import State, PlaybookEntry
from module.LLMs import gpt
from config.getenv import GetEnv
from utils import Logger, highlight_print

env = GetEnv()
llm = gpt()
logger = Logger(__name__)
json_parser = StrictJsonOutputParser()

# CHAINS
generator_chain = generator_prompt() | llm | json_parser
evaluator_chain = evaluator_prompt() | llm | json_parser
reflector_chain = reflector_prompt() | llm | json_parser
curator_chain = curator_prompt() | llm | json_parser

async def generator_node(state : State) -> State:
    logger.debug("GENERATOR")

    # Retrieved Playbook
    retrieved_bullets = state.get("retrieved_bullets", [])

    inputs = {
        "query" : state.get("query"),
        "retrieved_bullets" : retrieved_bullets
    }
    generation = await generator_chain.ainvoke(inputs)

    solution = generation.get("solution", "")

    # [uuid-123, uuid-456] 같은 playbook의 entry_id 값
    used_id = generation.get("used_bullet_ids", [])

    return {
        "solution" : solution,
        "trajectory" : [solution],
        "used_bullet_ids" : used_id
    }


async def evaluator_node(state : State) -> State:
    logger.debug("EVALUATOR")

    query = state.get("query")
    solution = state.get("solution")

    inputs = {
        "query" : query,
        "solution" : solution
    }

    # rating 과 comment 형식의 feedback
    feedback = await evaluator_chain.ainvoke(inputs)

    return {
        "feedback" : feedback
    }


async def reflector_node(state : State) -> State:
    logger.debug("REFLECTOR")

    used_bullets = [entry for entry in state['playbook'] if entry['entry_id'] in state.get("used_bullet_ids", [])]
    used_bullets_str = [f"[{entry['entry_id']}] {entry['content']}" for entry in used_bullets] or "No related items"

    query = state.get("query")
    trajectory = state.get("trajectory") # generator
    
    feedback = state.get("feedback", "") # evaluator

    inputs = {
        "query" : query,
        "trajectory" : trajectory,
        "used_bullets" : used_bullets_str,
        "feedback" : feedback
    }

    # root_cause, key_insight, bullet_tags 3개의 key 값을 가진 JSON 반환
    reflection = await reflector_chain.ainvoke(inputs)

    return {
        "reflection" : reflection
    }


async def curator_node(state : State) -> State:
    logger.debug("CURATOR")

    playbook_str = '\n'.join(f"[{entry['entry_id']}] {entry['content']}" for entry in state['playbook']) or ""

    reflection = state.get("reflection")

    inputs = {
        "playbook" : playbook_str,
        "reflection" : reflection
    }

    # reasoning, operations 2개의 key값을 가진 JSON 반환
    new_insights = await curator_chain.ainvoke(inputs)

    print(new_insights)

    operations = new_insights.get("operations", [])
    return {
        "new_insights" : operations
    }

