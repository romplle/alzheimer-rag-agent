import os
import streamlit as st

from llama_index.core import (
    Settings,
    StorageContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openrouter import OpenRouter

from api_key import OPENROUTER_API_KEY
from prompt import QA_PROMPT


DATA_FOLDER = 'data/extracted_texts'
PERSIST_DIR = 'data/index_storage'

DEVICE = 'cuda'

@st.cache_resource
def load_query_engine():
    Settings.embed_model = HuggingFaceEmbedding(
        model_name='Qwen/Qwen3-Embedding-0.6B',
        device=DEVICE
    )

    Settings.llm = OpenRouter(
        api_key=OPENROUTER_API_KEY,
        model='mistralai/devstral-2512:free',
        max_tokens=2000
    )

    if not os.path.exists(PERSIST_DIR):
        documents = SimpleDirectoryReader(input_dir=DATA_FOLDER, required_exts='.md').load_data(show_progress=True)
        splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=100)
        index = VectorStoreIndex.from_documents(documents, transformations=[splitter], show_progress=True)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)

    reranker = SentenceTransformerRerank(
        model='BAAI/bge-reranker-large',
        top_n=10,
        device=DEVICE
    )

    query_engine = index.as_query_engine(
        similarity_top_k=25,
        node_postprocessors=[reranker],
        response_mode='compact',
        text_qa_template=QA_PROMPT,
    )
    return query_engine
