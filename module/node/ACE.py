import os
from langchain_core.output_parsers import JsonOutputParser

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import uuid

from module.LLMs import gpt
