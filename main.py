from enum import Enum
from os import environ
import streamlit as st

st.set_page_config(
    page_title="ChatGLM3 Demo",
    page_icon=":robot:",
    layout='centered',
    initial_sidebar_state='expanded',
)

import demo_chat, demo_ci, demo_tool

# You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.
DEFAULT_SYSTEM_PROMPT = '''
你是一个由智谱AI训练的大语言模型，名字叫ChatGLM3，请和用户进行对话，仔细完成用户的命令。使用Markdown格式进行回复。
'''.strip()

# Set the title of the demo
st.title("ChatGLM3 Demo")

# Add your custom text here, with smaller font size
st.markdown("<sub>智谱AI 公开在线技术文档: https://lslfd0slxc.feishu.cn/wiki/WvQbwIJ9tiPAxGk8ywDck6yfnof </sub> \n\n <sub> 更多 ChatGLM3-6B 的使用方法请参考文档。</sub>", unsafe_allow_html=True)

class Mode(str, Enum):
    CHAT, TOOL, CI = '💬 Chat', '🛠️ Tool', '🧑‍💻 Code Interpreter'


with st.sidebar:
    top_p = st.slider(
        'top_p', 0.0, 1.0, 0.8, step=0.01
    )
    temperature = st.slider(
        'temperature', 0.0, 1.5, 0.95, step=0.01
    )
    repetition_penalty = st.slider(
        'repetition_penalty', 0.0, 2.0, 1.2, step=0.01
    )
    system_prompt = st.text_area(
        label="System Prompt (Only for chat mode)",
        height=300,
        value=DEFAULT_SYSTEM_PROMPT,
    )

prompt_text = st.chat_input(
    'Chat with ChatGLM3!',
    key='chat_input',
)

if not environ.get('USE_FASTLLM'):
    tab = st.radio(
        'Mode',
        [mode.value for mode in Mode],
        horizontal=True,
        label_visibility='hidden',
    )
else:
    tab = st.radio(
        'Mode',
        [mode.value for mode in Mode],
        key=Mode.CHAT,
        horizontal=True,
        label_visibility='hidden',
        disabled=True,
    )

if tab == Mode.CHAT:
    demo_chat.main(top_p, temperature, system_prompt, prompt_text, repetition_penalty)
elif tab == Mode.TOOL:
    demo_tool.main(top_p, temperature, prompt_text, repetition_penalty)
elif tab == Mode.CI:
    demo_ci.main(top_p, temperature, prompt_text, repetition_penalty)
else:
    st.error(f'Unexpected tab: {tab}')
