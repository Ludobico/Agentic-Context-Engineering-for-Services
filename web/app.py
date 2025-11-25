import streamlit as st
import requests
import json
import time

from config.getenv import GetEnv

env = GetEnv()
port = env.get_backend_config['BACKEND_PORT']

st.set_page_config(
    page_title="ACE Framework",
    layout='wide'
)

API_URL = f"http://localhost:{port}/chat/stream"

st.title("ACE Framework : Self-Improving Agent")
st.caption("Retrieval → Generation (Serving) | Evaluation → Reflector → Curator → Update")


# sidebar
with st.sidebar:
    st.header("System Status")
    st.info("Serving Graph : Active")
    st.info("Learning Graph : Background")

    st.divider()
    st.markdown("### Knowledge Graph")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if prompt := st.chat_input(""):
    st.session_state.messages.append({"role" : "user", "content" : prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        def response_generator():
            payload = {"query" : prompt}

            try:
                with requests.post(API_URL, json=payload, stream=True) as response:
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