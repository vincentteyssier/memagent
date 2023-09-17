import faiss
import os
from llama_index import Document
from dotenv import load_dotenv
load_dotenv()

FAISS_INDEX_PATH="faiss_index"

d = 768
faiss_index = faiss.IndexFlatL2(d)

from llama_index import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index import LangchainEmbedding, ServiceContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import set_global_service_context
from llama_index.llms import AzureOpenAI

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_BASE"] = os.getenv('OPENAI_API_BASE')
os.environ["OPENAI_API_TYPE"] = os.getenv('OPENAI_API_TYPE')
os.environ["OPENAI_API_VERSION"] = os.getenv('OPENAI_API_VERSION')

# initialize embeddings 
base_embeddings = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="thenlper/gte-base")
)

# initialize LLM
llm = AzureOpenAI(engine=os.getenv('OPENAI_DEPLOYMENT_NAME'), 
    model=os.getenv('OPENAI_MODEL_NAME'),
    temperature=0
)

# set global service context
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=base_embeddings,
    chunk_size=512
)
set_global_service_context(service_context)


documents=[Document(text="This is my initialization message")]

vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
index.storage_context.persist(persist_dir=FAISS_INDEX_PATH) 
