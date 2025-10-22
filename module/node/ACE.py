import os
from langchain_core.output_parsers import JsonOutputParser

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import uuid

from module.LLMs import gpt
from module.embed import EmbeddingPreprocessor
from module.prompt import curator_prompt, evaluator_prompt, generator_prompt, reflector_prompt
from module.state import State, PlaybookEntry

from utils import Logger, highlight_print

logger = Logger(__name__)
json_parser = JsonOutputParser()
llm = gpt()
embedding_model = EmbeddingPreprocessor.default_embedding_model()

generator_chain = generator_prompt() | llm | json_parser
reflector_chain = reflector_prompt() | llm | json_parser
curator_chain = curator_prompt() | llm | json_parser
evaluator_chain = evaluator_prompt() | llm | json_parser