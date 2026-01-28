from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)
from llama_index.llms.openrouter import OpenRouter

from api_key import OPENROUTER_API_KEY
from rag import load_query_engine


query_engine = load_query_engine()

evaluator_llm = OpenRouter(
    api_key=OPENROUTER_API_KEY,
    model='arcee-ai/trinity-large-preview:free',
)

faithfulness_evaluator = FaithfulnessEvaluator(llm=evaluator_llm)
relevancy_evaluator = RelevancyEvaluator(llm=evaluator_llm)

questions = [
    "What are potential targets for Alzheimer's disease treatment?",
    "Are the targets druggable with small molecules, biologics, or other modalities?",
    "What additional studies are needed to advance these targets?",

]

eval_results = []

for question in questions:
    response = query_engine.query(question)

    print('\nQuestion:', question)
    print(f'Answer:\n{response.response}\n')

    faithfulness_result = faithfulness_evaluator.evaluate_response(
        query=question,
        response=response
    )
    
    relevancy_result = relevancy_evaluator.evaluate_response(
        query=question,
        response=response
    )
    
    print(f'faithfulness: {faithfulness_result.score}')
    print(f'relevancy: {relevancy_result.score}')
    print(f'response_length: {len(response.response)}')
