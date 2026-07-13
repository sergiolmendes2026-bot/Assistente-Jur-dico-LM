import os
import tempfile
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Loaders específicos
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"
st.set_page_config(page_title="Assistente Jurídico LM", layout="wide")
st.markdown("## ⚖️ Assistente Jurídico LM")

with st.sidebar:
    api_key = st.text_input("GROQ API Key", type="password")

if not api_key:
    st.warning("Informe a GROQ API Key na barra lateral.")
    st.stop()
os.environ["GROQ_API_KEY"] = api_key

llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.2)

# Uploader ajustado para incluir Excel se desejar no futuro
uploaded_files = st.file_uploader(
    "Envie seus documentos (PDF, Word, TXT)", 
    type=["pdf", "docx", "txt"], 
    accept_multiple_files=True,
    key="uploader_v2"
)

if "chroma_dir" not in st.session_state:
    st.session_state.chroma_dir = tempfile.mkdtemp()

def processar_arquivos(files):
    docs = []
    for file in files:
        ext = os.path.splitext(file.name)[1].lower()
        # Cria um arquivo temporário persistente para o loader acessar
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(tmp_path)
            elif ext == ".docx":
                loader = Docx2txtLoader(tmp_path)
            elif ext == ".txt":
                loader = TextLoader(tmp_path, encoding='utf-8')
            else:
                continue
            
            docs.extend(loader.load())
        finally:
            # Opcional: remover o arquivo temporário após o carregamento
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-bert-base-dot-v5")
    return Chroma.from_documents(chunks, embeddings, persist_directory=st.session_state.chroma_dir)

if uploaded_files and "vectordb_ready" not in st.session_state:
    with st.spinner("Processando documentos..."):
        st.session_state.vectordb = processar_arquivos(uploaded_files)
        st.session_state.vectordb_ready = True
        st.success("Pronto! Pode fazer sua pergunta.")

if st.session_state.get("vectordb_ready"):
    retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 4})
    pergunta = st.text_area("Sua pergunta")
    
    if st.button("Perguntar"):
        rag_pipeline = RunnableParallel(
            context=(retriever | (lambda docs: "\n\n".join([d.page_content for d in docs]))),
            question=RunnablePassthrough()
        ) | ChatPromptTemplate.from_template("Responda com base no contexto: {context}\n\nPergunta: {question}") | llm | StrOutputParser()
        
        resposta = rag_pipeline.invoke(pergunta)
        st.markdown("### Resposta")
        st.write(resposta)
