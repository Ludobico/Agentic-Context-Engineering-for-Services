from core.state import PlaybookEntry
from langchain_core.documents import Document
from datetime import datetime
from module.db_management import PlayBookDB, VectorStore, get_db_instance, get_vector_store_instance
from node.nodes import update_playbook_node

import asyncio
from config.getenv import GetEnv

env = GetEnv()

async def test_delete_directly():
    db = get_db_instance()
    vector_store = get_vector_store_instance()

    entry_to_keep = PlaybookEntry(
        entry_id="test_keep_001",
        category="test",
        content="This is a helpful entry",
        helpful_count=3,
        harmful_count=1,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

    entry_to_prune = PlaybookEntry(
        entry_id="test_prune_001",
        category="test",
        content="This is a harmful entry",
        helpful_count=1,
        harmful_count=3,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

    initial_playbook = [entry_to_keep, entry_to_prune]

    print("STEP 1 : Adding test entries to DB and vecto store")
    db.add_entry(entry_to_keep)
    db.add_entry(entry_to_prune)

    docs_to_add = [
        Document(page_content=entry_to_keep['content'], metadata = {"entry_id" : entry_to_keep['entry_id']}),
        Document(page_content=entry_to_prune['content'], metadata = {"entry_id" : entry_to_prune['entry_id']}),
    ]
    vector_store.to_disk(docs_to_add)

    test_state = {
        "playbook" : initial_playbook,
        "reflection" : {},
        "new_insights" : []
    }

    print("STEP 2 : Running update_playbook node to trigger pruning")
    final_state = await update_playbook_node(test_state)

    print("STEP 3 : Verifying rsults")
    is_pruned_in_state = 'test_prune_001' not in [e['entry_id'] for e in final_state['playbook']]
    pruned_entry_id_db = db.get_entry("test_prune_001") is None

    await asyncio.sleep(1)
    vector_store_count_after = vector_store.get_doc_count()

    print("STEP 4 : Cleaning up test entries")
    db.delete_entry("test_keep_001")
    if not pruned_entry_id_db:
        db.delete_entry("test_prune_001")
    vector_store.delete_by_entry_ids(["test_keep_001", "test_prune_001"])

    print("TEST COMPLETE")

if __name__ == "__main__":
    asyncio.run(test_delete_directly())

