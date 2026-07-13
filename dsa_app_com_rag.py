import os
import tempfile
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Desativa avisos de paralelismo
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuração da página
st.set_page_config(page_title="Assistente Jurídico LM", layout="wide")
st.markdown("<div class='title'>⚖️ Assistente Jurídico LM</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Configurações")
    api_key = st.text_input("GROQ API Key", type="password")

# Verificação de API
if not api_key:
    st.warning("Informe a GROQ API Key na barra lateral.")
    st.stop()
os.environ["GROQ_API_KEY"] = api_key

# Inicializa o modelo
llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.2)

# Uploader de arquivos (agora aceita múltiplos formatos)
uploaded_files = st.file_uploader(
    "Envie seus documentos jurídicos", 
    type=["pdf", "docx", "txt", "xlsx", "csv"], 
    accept_multiple_files=True
)

if "chroma_dir" not in st.session_state:
    st.session_state.chroma_dir = tempfile.mkdtemp()

# Função de processamento universal
def dsa_cria_banco_vetorial(files):
    docs = []
    for file in files:
        # Cria arquivo temporário mantendo a extensão original
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.name}") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        
        # Unstructured detecta o formato automaticamente
        loader = UnstructuredFileLoader(tmp_path)
        docs.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-bert-base-dot-v5")
    return Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=st.session_state.chroma_dir)

# Gatilho de indexação
if uploaded_files and "vectordb_ready" not in st.session_state:
    with st.spinner("Processando documentos..."):
        st.session_state.vectordb = dsa_cria_banco_vetorial(uploaded_files)
        st.session_state.vectordb_ready = True
        st.success("Documentos prontos para consulta!")

# Pipeline RAG
if st.session_state.get("vectordb_ready"):
    retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 4})
    
    def formata_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    pergunta = st.text_area("Escreva sua pergunta jurídica")
    if st.button("Perguntar"):
        rag_pipeline = RunnableParallel(
            context=retriever | formata_docs,
            question=RunnablePassthrough()
        ) | ChatPromptTemplate.from_template(
            "Você é um assistente jurídico especializado. Responda com base no contexto abaixo:\n\n{context}\n\nPergunta: {question}"
        ) | llm | StrOutputParser()
        
        with st.spinner("Analisando..."):
            resposta = rag_pipeline.invoke(pergunta)
            st.markdown("### Resposta")
            st.write(resposta)
