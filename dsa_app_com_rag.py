import os
import tempfile
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredExcelLoader

# Configuração da Página
st.set_page_config(page_title="Assistente Jurídico LM", layout="wide")

# Barra Lateral - Design Solicitado
with st.sidebar:
    st.header("Configurações")
    api_key = st.text_input("Coloque aqui sua GROQ API Key e pressione Enter", type="password")
    st.markdown("---")
    st.header("Instruções")
    st.markdown("1. Informe sua chave.\n2. Faça o upload dos arquivos.\n3. Pergunte.")
    st.warning("Aviso: a IA pode cometer erros.Verifique fatos críticos")
        if st.button("📧 Clique Aqui Se Precisar de Suporte"):
        st.write("sergiolmendes2026@gmail.com")

if not api_key:
    st.markdown("# ⚖️ Assistente Jurídico")
    st.warning("Informe a GROQ API Key na barra lateral para continuar.")
    st.stop()

os.environ["GROQ_API_KEY"] = api_key
# Modelo atualizado e disponível
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

st.markdown("# ⚖️ Assistente Jurídico")
uploaded_files = st.file_uploader("Envie documentos (PDF, Word, TXT, Excel)", accept_multiple_files=True, type=["pdf", "docx", "txt", "xlsx"])

def processar_arquivos(arquivos):
    docs = []
    for f in arquivos:
        ext = os.path.splitext(f.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(f.read())
            path = tmp.name
        if ext == ".pdf": loader = PyPDFLoader(path)
        elif ext == ".docx": loader = Docx2txtLoader(path)
        elif ext == ".txt": loader = TextLoader(path, encoding='utf-8')
        elif ext == ".xlsx": loader = UnstructuredExcelLoader(path)
        else: continue
        docs.extend(loader.load())
    
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(chunks, embeddings)

if uploaded_files and st.button("Processar Arquivos"):
    with st.spinner("Indexando..."):
        st.session_state.db = processar_arquivos(uploaded_files)
        st.success("Arquivos prontos!")

if "db" in st.session_state and st.session_state.db:
    pergunta = st.text_area("Escreva sua pergunta jurídica", placeholder="Ex.: Quais cláusulas tratam de rescisão e multas?")
    if st.button("Perguntar"):
        retriever = st.session_state.db.as_retriever()
        template = "Responda baseando-se no contexto: {context}\n\nPergunta: {question}"
        chain = (
            {"context": retriever | (lambda d: "\n\n".join([doc.page_content for doc in d])), "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_template(template)
            | llm
            | StrOutputParser()
        )
        try:
            st.write(chain.invoke(pergunta))
        except Exception as e:
            st.error(f"Erro na consulta: {e}")
