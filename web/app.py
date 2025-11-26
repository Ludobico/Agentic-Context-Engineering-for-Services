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
st.caption("Retrieval → Generation (Serving) | Evaluation → Reflector → Curator → Update")

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

if "session_id" not in st.session_state:
    all_sessions = get_all_sessions()

    if all_sessions:
        st.session_state.session_id = all_sessions[0]
        st.session_state.is_new_chat = False
    
    else:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.is_new_chat = True
    
    st.session_state.messages = []

if not st.session_state.messages and not st.session_state.get("is_new_chat", False):
    history_data = get_history(st.session_state.session_id)

    if history_data:
        for msg in history_data:
            role = 'user' if msg['type'] == 'user' else 'assistant'
            st.session_state.messages.append({
                "role" : role,
                "content" : msg['content']
            })
    else:
        pass

# sidebar
with st.sidebar:
    if st.button("New Chat", use_container_width=True, type='primary'):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.is_new_chat = True
        st.rerun()
    
    st.divider()
    all_sessions = get_all_sessions()
    if not all_sessions:
        st.caption("No history yet")
    
    for sid in all_sessions:
        is_current = (sid == st.session_state.session_id)

        label = f"Chat {sid[:8]}"
        if is_current:
            label += "(Current)"
        
        if st.button(label, key=sid, use_container_width=True, disabled=is_current):
            st.session_state.session_id = sid
            st.session_state.messages = []
            st.session_state.is_new_chat = False
            st.rerun()

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

if "messages" not in st.session_state:
    st.session_state.messages = []

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