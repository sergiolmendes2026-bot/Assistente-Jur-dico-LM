import os
import tempfile
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_community.document_loaders import UnstructuredFileLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

os.environ["TOKENIZERS_PARALLELISM"] = "false"
st.set_page_config(page_title="Assistente Jurídico Multiformato", layout="wide")

st.title("⚖️ Assistente Jurídico Multiformato")

with st.sidebar:
    api_key = st.text_input("GROQ API Key", type="password")
    tipo_input = st.radio("Fonte de Dados", ["Arquivos (PDF, Word, TXT, Excel)", "Website (URL)"])
    
    arquivos_input = None
    if tipo_input == "Arquivos (PDF, Word, TXT, Excel)":
        arquivos_input = st.file_uploader("Suba seus documentos", accept_multiple_files=True)
    else:
        url_input = st.text_input("Cole a URL do site")

if not api_key:
    st.warning("Informe a API Key.")
    st.stop()

os.environ["GROQ_API_KEY"] = api_key
llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.2)

if "chroma_dir" not in st.session_state:
    st.session_state.chroma_dir = tempfile.mkdtemp()

def processar_documentos(input_data, tipo):
    docs = []
    if tipo == "site":
        loader = WebBaseLoader(input_data)
        docs.extend(loader.load())
    else:
        for file in input_data:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.name}") as tmp:
                tmp.write(file.read())
                loader = UnstructuredFileLoader(tmp.name)
                docs.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-bert-base-dot-v5")
    return Chroma.from_documents(chunks, embeddings, persist_directory=st.session_state.chroma_dir)

if st.button("Processar Documentos"):
    with st.spinner("Processando..."):
        if tipo_input == "Arquivos (PDF, Word, TXT, Excel)" and arquivos_input:
            st.session_state.vectordb = processar_documentos(arquivos_input, "arquivos")
            st.session_state.vectordb_ready = True
        elif tipo_input == "Website (URL)" and url_input:
            st.session_state.vectordb = processar_documentos(url_input, "site")
            st.session_state.vectordb_ready = True
        st.success("Pronto!")

# Pipeline RAG
if st.session_state.get("vectordb_ready"):
    retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 4})
    
    def formata_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    pergunta = st.text_area("Sua pergunta jurídica")
    if st.button("Enviar"):
        rag_pipeline = RunnableParallel(
            context=retriever | formata_docs,
            question=RunnablePassthrough()
        ) | ChatPromptTemplate.from_template("Responda com base no contexto: {context}\n\nPergunta: {question}") | llm | StrOutputParser()
        
        resposta = rag_pipeline.invoke(pergunta)
        st.markdown("### Resposta")
        st.write(resposta)
