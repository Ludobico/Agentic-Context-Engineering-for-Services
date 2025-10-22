from typing import TypedDict, Optional, List, Dict

class PlaybookEntry(TypedDict):
    entry_id : str
    category : str
    content : str
    helpful_count : int
    harmful_count : int
    created_step : int
    last_used_step : int

class State(TypedDict):
    query : str
    playbook : List[PlaybookEntry]
    solution : Optional[str]
    verbose : Optional[bool]

    retrieved_bullets : List[PlaybookEntry]
    used_bullet_ids : List[str]

    # learning state
    trajectory : List[str]
    reflection : Optional[Dict]
    new_insights : Optional[List[Dict]]
    feedback : Optional[Dict]

    # playnook management
    current_step : int

    max_playbook_size : int
    dedup_threshold : float
    retrieval_threshhold : float

    