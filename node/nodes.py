from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from module.prompt import curator_prompt, evaluator_prompt, generator_prompt, reflector_prompt
from node.node_utils import SolutionOnlyStreamCallback
from core import State, PlaybookEntry
from module.LLMs import gpt
from config.getenv import GetEnv
from utils import Logger, highlight_print

env = GetEnv()
llm = gpt()
logger = Logger(__name__)
json_parser = JsonOutputParser()

# CHAINS
generator_chain = generator_prompt() | llm | json_parser

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
    used_id = generation.get("used_bullet_ids", [])

    logger.debug(f"used_id : {used_id}")

    return {
        "solution" : solution,
        "trajectory" : [solution],
        "used_bullet_ids" : used_id
    }



    
