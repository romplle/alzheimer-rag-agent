from pathlib import Path
import streamlit as st

from rag import load_query_engine

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
