import streamlit as st
import requests
import json
import time
import uuid

from config.getenv import GetEnv

env = GetEnv()
port = env.get_backend_config['BACKEND_PORT']

st.set_page_config(
    page_title="ACE Framework",
    layout='wide'
)

API_URL = f"http://localhost:{port}"

st.title("ACE Framework : Self-Improving Agent")

# session management
def get_all_sessions():
    res = requests.get(f"{API_URL}/chat/sessions")
    if res.status_code == 200:
        return res.json().get("sessions", [])
    return []

def get_history(sid):
    res = requests.get(f"{API_URL}/chat/history/{sid}")
    if res.status_code == 200:
        return res.json()
    return []

def get_chat_history(sid : str):
    res = requests.get(f"{API_URL}/chat/history/{sid}")
    if res.status_code == 200:
        history_data = res.json()

        messages = []
        for msg in history_data:
            role = 'user'if msg['type'] == 'user' else 'assistant'
            messages.append({
                "role" : role,
                "content" : msg['content']
            })
        return messages
    
    return []

def init_new_chat():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.is_new_chat = True

if "session_id" not in st.session_state:
    existing_sessions = get_all_sessions()

    if existing_sessions:
        st.session_state.session_id = existing_sessions[0]
        st.session_state.messages = []
        st.session_state.is_new_chat = False
    
    else:
        init_new_chat()


if not st.session_state.messages and not st.session_state.get("is_new_chat", False):
    loaded_messages = get_chat_history(st.session_state.session_id)

    if loaded_messages:
        st.session_state.messages = loaded_messages
    else:
        pass


# sidebar
# ---------------------------------------------------------
with st.sidebar:
    # new chat
    if st.button("New Chat", use_container_width=True, type='secondary'):
        init_new_chat()
        st.rerun()

    # delete chat
    if st.session_state.session_id and not st.session_state.get("is_new_chat", False):
        if st.button("Delete Current Chat", use_container_width=True, type='primary'):
            del_res = requests.delete(f"{API_URL}/chat/history/{st.session_state.session_id}")

            if del_res.status_code == 200:
                st.success("Chat deleted successfully")
                time.sleep(0.5)
                init_new_chat()
                st.rerun()
    
    st.caption(f"Current Session : {st.session_state.session_id[:8]}")
    st.divider()

    st.subheader("Chat History")
    sessions = get_all_sessions()

    if not sessions:
        pass

    else:
        for sid in sessions:
            is_current = (sid == st.session_state.session_id)

            label = f"Chat {sid[:8]}"
            if is_current:
                label += "(Current)"
            
            if st.button(label, key=sid, use_container_width=True, disabled=is_current):
                st.session_state.session_id = sid
                st.session_state.messages = []
                st.session_state.is_new_chat = False
                st.rerun()
    
    st.divider()

    st.header("Model Settings")
    valid_providers = {}

    if env.get_openai_api_key and env.get_openai_api_key.strip():
        valid_providers['OpenAI'] = 'openai'
    
    if env.get_claude_api_key and env.get_claude_api_key.strip():
        valid_providers["Anthropic"] = "anthropic"

    if env.get_gemini_api_key and env.get_gemini_api_key.strip():
        valid_providers["Google"] = "google"
    
    if not valid_providers:
        st.error("No active API Keys found")
        st.warning("Please set at least one API Key in `config.ini`")
    
    selected_provider_label = st.selectbox(
        "Provider",
        options=list(valid_providers.keys()),
        index=0
    )

    provider_id = valid_providers[selected_provider_label]

    model_options = []
    if provider_id == 'openai':
        model_options = [env.get_openai_model]
    
    elif provider_id == "anthropic":
        model_options = [env.get_claude_model]
    
    elif provider_id == "google":
        model_options = [env.get_gemini_model]

    selected_model = st.selectbox(
        "Model",
        options=model_options,
        index=0,
        disabled=True
    )

    st.info(f"Using: **{selected_model}**")
    st.caption("To change the model, update `config.ini`.")
    st.divider()
# ---------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if prompt := st.chat_input(""):
    st.session_state.messages.append({"role" : "user", "content" : prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.is_new_chat = False
    
    with st.chat_message("assistant"):
        def response_generator():
            payload = {
                "query" : prompt,
                "llm_provider" : provider_id,
                "llm_model" : selected_model,
                "session_id" : st.session_state.session_id
                }

            try:
                with requests.post(f"{API_URL}/chat/stream", json=payload, stream=True) as response:
                    if response.status_code != 200:
                        yield f"Error : Server returned status {response.status_code}"
                    
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')

                            if decoded_line.startswith("data: "):
                                data_str = decoded_line[6:]

                                if data_str == "[DONE]":
                                    break

                                try:
                                    json_data = json.loads(data_str)
                                    token = json_data.get("token", "")
                                    yield token
                                
                                except json.JSONDecodeError:
                                    pass
            except requests.exceptions.ConnectionError:
                yield "Server Error"

        full_response = st.write_stream(response_generator())
    
    st.session_state.messages.append({"role" : "assistant", "content" : full_response})

if (st.session_state.messages) == 2:
    st.rerun()