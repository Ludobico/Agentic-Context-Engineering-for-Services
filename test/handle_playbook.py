from core.state import PlaybookEntry
from langchain_core.documents import Document
from datetime import datetime
from module.db_management import PlayBookDB, VectorStore, get_db_instance, get_vector_store_instance
from node.nodes import update_playbook_node

import asyncio
from config.getenv import GetEnv

env = GetEnv()
db = get_db_instance()
vector_store = get_vector_store_instance()

async def test_delete_directly():

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
    pruned_entry_in_db = db.get_entry("test_prune_001") is None

    await asyncio.sleep(1)
    vector_store_count_after = vector_store.get_doc_count()
    print(f"  - Pruned from state? {'✅ Yes' if is_pruned_in_state else '❌ No'}")
    print(f"  - Pruned from DB? {'✅ Yes' if pruned_entry_in_db else '❌ No'}")
    print(f"Vector_store count after pruning : {vector_store_count_after}")

    print("STEP 4 : Cleaning up test entries")
    db.delete_entry("test_keep_001")
    if not pruned_entry_in_db:
        db.delete_entry("test_prune_001")
    vector_store.delete_by_entry_ids(["test_keep_001", "test_prune_001"])

    print("TEST COMPLETE")

async def test_update_directly():

    entry_id_to_update = "test_update_001"
    original_content = "This is the original content"
    updated_content = "This is the NEW and UPDATED content"

    initial_entry = PlaybookEntry(
        entry_id=entry_id_to_update,
        category="test",
        content=original_content,
        helpful_count=1,
        harmful_count=0,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    print("STEP 1: Adding initial entry...")
    doc = Document(
        page_content=initial_entry['content'],
        metadata={
            "entry_id" : initial_entry['entry_id']
        }
    )
    db.add_entry(initial_entry)
    vector_store.to_disk([doc])

    update_operation = {
        "type" : "UPDATE",
        "entry_id" : entry_id_to_update,
        "content" : updated_content
    }

    test_state = {
        "playbook" : [initial_entry],
        "reflection" : {},
        "new_insights" : [update_operation]
    }

    print("\nSTEP 2: Running update_playbook_node to trigger UPDATE...")
    await update_playbook_node(test_state)

    print("STEP 3: Verifying results...")
    db_entry_after = db.get_entry(entry_id_to_update)
    is_db_updated = db_entry_after['content'] == updated_content
    print(f"  - Content updated in DB? {'✅ Yes' if is_db_updated else '❌ No'}")

    vs_payload_after = vector_store.get_entry_by_id(entry_id_to_update)
    is_vs_updated = vs_payload_after and vs_payload_after.get("page_content") == updated_content
    print(f"  - Content updated in Vector Store? {'✅ Yes' if is_vs_updated else '❌ No'}")

    print("\nSTEP 4: Cleaning up test entry...")
    db.delete_entry(entry_id_to_update)
    vector_store.delete_by_entry_ids([entry_id_to_update])


if __name__ == "__main__":
    # asyncio.run(test_delete_directly())
    asyncio.run(test_update_directly())
    payload = vector_store.get_all_entries()
    print(payload)