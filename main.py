import streamlit as st 
import tempfile 
import os
import torch 
from langchain_community.document_loaders import PyPDFLoader 
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_huggingface.llms import HuggingFacePipeline
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None 
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "embeddings" not in st.session_state:
    st.embeddings = None 
if "llm" not in st.session_state:
    st.llm = None 


@st.cache_resource 
def load_embeddings():
    return HuggingFaceEmbeddings(model_name = "bkai-foundation-models/vietnamese-bi-encoder")

@st.cache_resource
def load_llm():
    MODEL_NAME  =  "lmsys/vicuna-7b-v1.5"
    nf4_config = BitsAndBytesConfig(
        load_in_4bit =True ,
        bnb_4bit_quant_type = "nf4",
        bnb_bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype = torch.bfloat16,
        low_cpu_mem_usage = True
    )

    token = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_pipeline = pipeline(
        "text-generation",
        model = model,
        tokenizer = tokenizer,
        max_new_tokens = 512,
        pad_token_id =tokenizer.eos_token_id,
        device_map = "auto"
    )
    return HuggingFacePipeline(pipeline= model_pipeline)
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    
    senmantic_splitter = SemanticChunker(
        embeddings=st.session_state.embeddings,
        bufer_size=1,
        breakpoin_threshold_type="percentile",
        breakpoin_threshold_amount=0.5,
        min_chunk_size=500,
        add_start_index=True
    )

    docs = senmantic_splitter.split_documents(documents)
    vector_db = Chroma.from_documents(documents=docs, embedding=st.session_state.embeddings)
    

    retriever = vector_db.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | st.session_state.llm
        | StrOutputParser()
    )
    os.unlink(tmp_file_path)
    return rag_chain, len(docs)

st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
st.title("PDF RAG Assistant")

st.markdown("""
 **Ứng dụng AI giúp bạn hỏi đáp trực tiếp với nội dung tài liệu PDF bằng tiếng Việt**
 **Cách sử dụng đơn giản:**
 1. **Upload PDF** Chọn file PDF từ máy tính và nhấn "Xử lý PDF"
 2. **Đặt câu hỏi** Nhập câu hỏi về nội dung tài liệu và nhận câu trả lời ngay lập tức
 ---
""")

if not st.session_state.models_loaded:
    with st.spinner("Đang tải mô hình..."):
        st.session_state.embeddings = load_embeddings()
        st.session_state.llm = load_llm()
        st.session_state.models_loaded = True
        st.success("Mô hình đã được tải thành công!")
        st.rerun()


uploaded_file = st.file_uploader("Chọn file PDF", type=["pdf"])
if uploaded_file and st.button("Xử lý PDF"):
    with st.spinner("Đang xử lý PDF..."):
        try:
            st.session_state.rag_chain, num_chunks = process_pdf(uploaded_file)
            st.success(f"Đã xử lý thành công {num_chunks} đoạn văn bản từ PDF!")
        except Exception as e:
            st.error(f"Có lỗi xảy ra: {e}")

if st.session_state.rag_chain:
    question = st.text_input("Nhập câu hỏi của bạn:")
    if question and st.button("Hỏi"):
        with st.spinner("Đang trả lời..."):
            try:
                answer = st.session_state.rag_chain.invoke({"question": question})
                st.success(f"Câu trả lời: {answer}")
            except Exception as e:
                st.error(f"Có lỗi xảy ra khi trả lời: {e}")