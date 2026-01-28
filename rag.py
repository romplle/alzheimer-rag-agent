import streamlit as st
import chromadb

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
from llama_index.vector_stores.chroma import ChromaVectorStore

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
        model='arcee-ai/trinity-large-preview:free',
        max_tokens=2000
    )

    chroma_client = chromadb.PersistentClient(
        path=PERSIST_DIR,
        settings=chromadb.Settings(anonymized_telemetry=False)
    )
    chroma_collection = chroma_client.get_or_create_collection('alzheimer_papers')
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR, vector_store=vector_store)

    if chroma_collection.count() == 0:
        documents = SimpleDirectoryReader(input_dir=DATA_FOLDER,required_exts=['.md']).load_data(show_progress=True)
        splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=100)
        index = VectorStoreIndex.from_documents(documents, transformations=[splitter], storage_context=storage_context, show_progress=True)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
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

if __name__ == '__main__':
    load_query_engine()
