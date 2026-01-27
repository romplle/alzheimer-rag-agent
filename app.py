import streamlit as st
from pathlib import Path

from llama_index.core import (
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openrouter import OpenRouter

from api_key import OPENROUTER_API_KEY
from prompt import QA_PROMPT


PERSIST_DIR = 'data/index_storage'
EMBED_MODEL = 'Qwen/Qwen3-Embedding-0.6B'
LLM_MODEL = 'mistralai/devstral-2512:free'

@st.cache_resource
def load_query_engine():
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        device='cuda'
    )

    Settings.llm = OpenRouter(
        api_key=OPENROUTER_API_KEY,
        model=LLM_MODEL,
        max_tokens=2000
    )

    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-large",
        top_n=10,
        device="cuda"
    )

    query_engine = index.as_query_engine(
        similarity_top_k=25,
        node_postprocessors=[reranker],
        response_mode="compact",
        text_qa_template=QA_PROMPT,
    )
    return query_engine


st.set_page_config(
    page_title='Alzheimer RAG-agent',
    layout='wide'
)

st.title('Alzheimer RAG-agent')

question = st.text_area(
    label='Question:',
    height=100,
    key='question_input'
)

if st.button('Ask', type='primary', use_container_width=True):
    query_engine = load_query_engine()
    response = query_engine.query(question)

    st.subheader('Answer')
    st.markdown(response.response)

    st.subheader('Sources')
    for i, node in enumerate(response.source_nodes, 1):
        file_name = Path(node.metadata.get('file_name', 'unknown')).name
        score = node.score
        preview = node.text[:250].replace('\n', ' ').strip()

        with st.expander(f'Source {i}: {file_name} (score: {score:.3f})'):
            st.text(preview)
