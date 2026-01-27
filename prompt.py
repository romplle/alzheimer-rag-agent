from llama_index.core import PromptTemplate

QA_PROMPT_TMPL = '''\
You are an expert on therapeutic targets for Alzheimer's disease.
Answer accurately, scientifically, based solely on the context provided.
If there is not enough information, be honest about it.

The context from the articles:
{context_str}

Question:
{query_str}

Response (indicating sources where possible):
'''

QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)
