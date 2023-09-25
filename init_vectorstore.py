import faiss
import os, logging, sys
from llama_index import Document
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

FAISS_INDEX_PATH="./storage"

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


documents=[
    Document(text="This is my initialization message"),
    Document(text="This is my initialization message1"),
    Document(text="This is my initialization message2"),
    Document(text="This is my initialization message3"),
    Document(text="This is my initialization message4"),
    Document(text="This is my initialization message5"),
    Document(text="This is my initialization message6"),
    Document(text="This is my initialization message7"),
    Document(text="This is my initialization message8"),
    Document(text="This is my initialization message9"),
    Document(text="This is my initialization message10"),
]

vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
index.storage_context.persist(persist_dir=FAISS_INDEX_PATH) 


vector_store = FaissVectorStore.from_persist_dir("./storage")
storage_context = StorageContext.from_defaults(
    vector_store=vector_store, persist_dir="./storage"
)
index = load_index_from_storage(storage_context=storage_context, service_context=service_context)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
display(Markdown(f"<b>{response}</b>"))

