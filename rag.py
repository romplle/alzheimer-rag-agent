import os

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openrouter import OpenRouter

from api_key import OPENROUTER_API_KEY
from prompt import QA_PROMPT


DATA_FOLDER = 'data/extracted_texts'
PERSIST_DIR = 'data/index_storage'

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

EMBED_MODEL = 'Qwen/Qwen3-Embedding-0.6B'
LLM_MODEL = 'mistralai/devstral-2512:free'

Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL,
    device='cuda',
)

Settings.llm = OpenRouter(
    api_key=OPENROUTER_API_KEY,
    model=LLM_MODEL,
    max_tokens=2000
)

if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader(input_dir=DATA_FOLDER, required_exts='.md').load_data(show_progress=True)
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    index = VectorStoreIndex.from_documents(documents, transformations=[splitter], show_progress=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
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

def ask(question: str):
    print('\nQuestion:', question)
    response = query_engine.query(question)

    print('\nAnswer:')
    print(response.response)


questions = [
    "What are the most promising therapeutic targets for Alzheimer's disease mentioned in recent studies?",
    "Is BACE1 still considered a valid drug target?",
    "What role does neuroinflammation play in Alzheimer's?",
]

for q in questions:
    ask(q)
