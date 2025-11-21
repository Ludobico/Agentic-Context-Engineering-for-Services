from typing import TypedDict, Optional, NotRequired
from datetime import datetime

class PlaybookEntry(TypedDict):
    entry_id : str
    category : str
    content : str
    helpful_count : int
    harmful_count : int
    
    # timestamp
    created_at : datetime
    updated_at : datetime
    # python 3.11 에서 NotRequired 가 사용되며, 키값 자체가 존재하지 않을수도있음을 의미
    last_used_at : NotRequired[datetime]

class State(TypedDict):
    query : str
    playbook : list[PlaybookEntry]
    solution : Optional[str]
    verbose : Optional[bool]

    retrieved_bullets : list[PlaybookEntry]
    used_bullet_ids : list[str]

    # learning state
    trajectory : list[str]
    reflection : Optional[dict]
    new_insights : Optional[list[dict]]
    feedback : Optional[dict]

    max_playbook_size : int
    dedup_threshold : float
    retrieval_threshold : float
    retrieval_topk : int

    # evaluation state
    test_code : NotRequired[str]
    test_id : NotRequired[str]
    ground_truth : NotRequired[str]

    