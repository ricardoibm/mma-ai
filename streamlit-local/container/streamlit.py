import streamlit as st
import requests
import os
from pymilvus import connections, utility
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import httpx
import json
import asyncio
import urllib.parse

# Streamlit app title
st.title("Retrieval Augmented Generation based on a given pdf")

# Milvus and LLAMA connection parameters
MILVUS_HOST = "milvus-service"
MILVUS_PORT = "19530"
LLAMA_HOST = "llama-service"
LLAMA_PORT = "8080"

# Function to download and process PDFs
@st.cache_resource
def load_and_process_pdfs():
    pdf_urls = [
        os.getenv("PDF_URL")
    ]
    # Parse the URL
    url_parts = urllib.parse.urlparse(os.getenv("PDF_URL"))
    # Get the path
    path_query = url_parts.path
    # Split the path into path and filename
    path_filename = os.path.split(path_query)
    # Get the filename
    pdf_names = [os.path.basename(path_filename[1])]

    all_docs = []
    
    for url, name in zip(pdf_urls, pdf_names):
        if not os.path.exists(name):
            output_path = os.path.join("/tmp/", name)
            st.write(f"Downloading {name}...")
            res = requests.get(url)
            with open(output_path, 'wb') as file:
                file.write(res.content)
        
        st.write(f"Processing {name}...")
        loader = PyPDFLoader(output_path)
        docs = loader.load()
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=768, chunk_overlap=0)
        split_docs = text_splitter.split_documents(docs)
        all_docs.extend(split_docs)
    
    st.write("Embedding documents...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="/work/", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    
    st.write("Connecting to Milvus...")
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    colls = utility.list_collections()
    for coll in colls:
        utility.drop_collection(coll)
    
    st.write("Creating vector store...")
    vector_store = Milvus.from_documents(
        all_docs,
        embedding=embeddings,
        collection_name="lighthouse",
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT}
    )
    
    st.write("Processing complete!")
    return vector_store

# Function to build prompt
def build_prompt(question, topn_chunks: list[str]):
    prompt = "Instructions: Compose a concise answer to the query using the provided search results, no need to mention you found it in the resuts\n\n"
    prompt += "Search results:\n"
    for chunk in topn_chunks:
        prompt += f"[Document: {chunk[0].metadata.get('source', 'Unknown')}, Page: {chunk[0].metadata.get('page', 'Unknown')}]: " + chunk[0].page_content.replace("\n", " ") + "\n\n"
    prompt += f"Query: {question}\n\nAnswer: "
    return prompt

# Asynchronous function to get LLAMA response
async def get_llama_response(prompt):
    json_data = {
        'prompt': prompt,
        'temperature': 0.1,
        'n_predict': 200,
        'stream': True,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream('POST', f'http://{LLAMA_HOST}:{LLAMA_PORT}/completion', json=json_data) as response:
            full_response = ""
            async for chunk in response.aiter_bytes():
                try:
                    data = json.loads(chunk.decode('utf-8')[6:])
                    if data['stop'] is False:
                        full_response += data['content']
                except:
                    pass
    return full_response

# Load and process PDFs
with st.spinner("Loading and processing PDFs... This may take a few minutes."):
    vector_store = load_and_process_pdfs()

# User input
question = st.text_input("Enter your question about the pdf you picked:")

if question:
    # Perform similarity search
    docs = vector_store.similarity_search_with_score(question, k=3)
    
    # Build prompt
    prompt = build_prompt(question, docs)
    
    # Get LLAMA response
    with st.spinner("Generating answer..."):
        answer = asyncio.run(get_llama_response(prompt))
    
    # Display answer
    st.write("Answer:", answer)
