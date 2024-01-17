from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain_openai import OpenAI
import streamlit as st
import pandas as pd
import os
import openai

file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}


def clear_submit():
    st.session_state["submit"] = False


@st.cache_data(ttl="2h")
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"Format non supporté: {ext}")
        return None


st.set_page_config(page_title="Filebot: Discutez avec vos données", page_icon="🦜")
st.title("Filebot: Discutez avec vos données")

uploaded_file = st.file_uploader(
    "Uploader un fichier",
    type=list(file_formats.keys()),
    help="Les formats excel et csv snt supportés",
    on_change=clear_submit,
)

if not uploaded_file:
    st.warning(
        "Cette application utilise l'outil PythonAstREPLTool de LangChain, qui est vulnérable à l'exécution arbitraire de code. Veuillez faire preuve de prudence lors du déploiement et du partage de cette application."
    )

if uploaded_file:
    df = load_data(uploaded_file)

openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
)
if "messages" not in st.session_state or st.sidebar.button(
    "Nettoyer l'historique de conversation"
):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]
    st.session_state["messages_fr"] = [
        {"role": "assistant", "content": "Comment puis-je vous aider?"}
    ]

for msg in st.session_state.messages_fr:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Des questions sur vos données?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages_fr.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    client = openai.OpenAI(api_key=openai_api_key)
    if not openai_api_key:
        st.info("Ajoutez une clef d'API OpenAI pour continuer.")
        st.stop()

    llm = OpenAI(
        openai_api_key=openai_api_key,
        max_tokens=256,
    )

    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])

        with st.spinner("Je finalise.."):
            messages = [
                {
                    "role": "system",
                    "content": "Tu traduis en français.",
                },
                {
                    "role": "user",
                    "content": "\r\n".join(response),
                },
            ]

            response_fr = (
                client.chat.completions.create(
                    model="gpt-3.5-turbo-16k",
                    messages=messages,
                    temperature=0.15,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                .choices[0]
                .message.content
            )
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.messages_fr.append(
            {"role": "assistant", "content": response_fr}
        )

        st.write(response_fr)
