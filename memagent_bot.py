import logging, os, sys, time
from csv import DictWriter
from dotenv import load_dotenv
from multiprocessing import Lock
from multiprocessing.managers import BaseManager

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, ContextTypes, CommandHandler, MessageHandler, filters, CallbackQueryHandler
from telegram.constants import ParseMode

# initialize the logger
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# LlamaIndex LLM/VectorStore/embeddings
import faiss
from langchain import OpenAI
from langchain.llms import AzureOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext
from local_embeddings import LocalHuggingFaceEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding
from llama_index import set_global_service_context


from llama_index import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.node_parser.extractors import (
    MetadataExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
)
from llama_index import Document
from llama_index.schema import TextNode
from llama_index.text_splitter import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore


load_dotenv()

FAISS_INDEX_PATH="faiss_index"
# list of column names
FILE_STORAGE_FIELD_NAMES = ['TIMESTAMP', 'TEXT']
FILE_STORAGE_PATH="./documents/history.csv"


# initialize the text splitter
text_splitter = SentenceSplitter(
    chunk_size=512,
    # separator=" ",
)

# load documents
#documents = SimpleDirectoryReader("../paul_graham_essay/data").load_data()

#storage_context = StorageContext.from_defaults(vector_store=vector_store)
#index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# save index to disk
#index.storage_context.persist()


# load index from disk
#vector_store = FaissVectorStore.from_persist_dir("./storage")
#storage_context = StorageContext.from_defaults(
#    vector_store=vector_store, persist_dir="./storage"
#)
#index = load_index_from_storage(storage_context=storage_context)

# set Logging to DEBUG for more detailed outputs
#query_engine = index.as_query_engine()
#response = query_engine.query("What did the author do growing up?")
#print(f"<b>{response}</b>")
# set Logging to DEBUG for more detailed outputs
#query_engine = index.as_query_engine()
#response = query_engine.query("What did the author do after his time at Y Combinator?")
#print(f"<b>{response}</b>")







# initialize LLM
#llm = AzureOpenAI(
#    openai_api_type=os.getenv('OPENAI_API_TYPE'),
#    openai_api_version=os.getenv('OPENAI_API_VERSION'),
#    openai_api_base=os.getenv('OPENAI_API_BASE'),
#    openai_api_key=os.getenv('OPENAI_API_KEY'),
#    deployment_name=os.getenv('DEPLOYMENT_NAME'), 
#    model_name=os.getenv('MODEL_NAME'), 
#    temperature = 0
#)

# initialize embeddings 
base_embeddings = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="thenlper/gte-base")
)
#service_context = ServiceContext.from_defaults(embed_model=base_embeddings, chunk_size=512)
#set_global_service_context(service_context)

# initialize the vector store
# dimensions of text-ada-embedding-002
#d = 1536
#faiss_index = faiss.IndexFlatL2(d)
#vector_store = FaissVectorStore.from_persist_dir(FAISS_INDEX_PATH)
#storage_context = StorageContext.from_defaults(
#    vector_store=vector_store, persist_dir=FAISS_INDEX_PATH
#)
#index = load_index_from_storage(storage_context=storage_context, service_context=service_context)

# load database in memory
#db = FAISS.load_local(FAISS_INDEX_PATH, base_embeddings)

# Store bot action status: store or retrieve
action = ""

# Pre-assign menu text
FIRST_MENU = "<b>Menu</b>\n\nChoose your action:"

# Pre-assign prompts
STORE_PROMPT = "Enter the notes to store:"
RETRIEVE_PROMPT = "Enter your question:"

# Pre-assign button text
STORE_BUTTON = "Store Note"
RETRIEVE_BUTTON = "Retrieve Memory"


# Build keyboards
FIRST_MENU_MARKUP = InlineKeyboardMarkup([[
    InlineKeyboardButton(STORE_BUTTON, callback_data=STORE_BUTTON),
    InlineKeyboardButton(RETRIEVE_BUTTON, callback_data=RETRIEVE_BUTTON)
]])



async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    This function would be added to the application as a handler for messages coming from the Bot API
    If global variable action is empty, display menu
    If global variable action is store, call store function
    If global variable action is retrieve, call retrieve function
    """

    # Print to console
    print(f'{update.message.from_user.first_name} wrote {update.message.text}')

    global action

    # return acknowledgement message
    if (action=="store"):
        print("Store")
        await store(update, context)
        # reset the action flag
        action=""
    elif (action=="retrieve"):
        print("Retrieve")
        #reset the action flag
        await retrieve(update, context)
        action=""
    else:
        print("Reset")
    
    # show back the main menu
    await context.bot.send_message(
        update.message.from_user.id,
        FIRST_MENU,
        parse_mode=ParseMode.HTML,
        reply_markup=FIRST_MENU_MARKUP
    )


async def menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    This handler sends a menu with the inline buttons we pre-assigned above
    """

    await context.bot.send_message(
        update.message.from_user.id,
        FIRST_MENU,
        parse_mode=ParseMode.HTML,
        reply_markup=FIRST_MENU_MARKUP
    )

async def store(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    This handler processes the user input by storing it in the vector db
    """

    # Print to console
    print(f'{update.message.from_user.first_name} wants to store: {update.message.text}')

    # split the message into chunks in case it is too big
    text_chunks = []
    doc_idxs = []
    for doc_idx, doc in enumerate([Document(text=update.message.text)]):
        cur_text_chunks = text_splitter.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))
    # create nodes that store metadata and model relationship between chunks
    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
        )
        src_doc = [Document(text=update.message.text)][doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)

    #metadata_extractor = MetadataExtractor(
    #    extractors=[
    #        TitleExtractor(nodes=5, llm=llm),
    #        QuestionsAnsweredExtractor(questions=3, llm=llm),
    #    ],
    #    in_place=False,
    #)
    #nodes = metadata_extractor.process_nodes(nodes)
    print("Nodes:")
    print(nodes)

    # generate embeddings for each node
    #for node in nodes:
    #    node_embedding = base_embeddings.get_text_embedding(
    #        node.get_content(metadata_mode="all")
    #    )
    #    node.embedding = node_embedding
    
    # print a sample node
    #print(nodes[0].get_content(metadata_mode="all"))

    # save the message with metadata on disk first for future re-indexing needs
    #with open(FILE_STORAGE_PATH, 'a') as f_object:
    #    row = {'TIMESTAMP': time.time(), 'TEXT': update.message.text}
    #    dictwriter_object = DictWriter(f_object, fieldnames=FILE_STORAGE_FIELD_NAMES)
    #    dictwriter_object.writerow(row)
    #    f_object.close()

    # index the message in the vector db
    #vector_store.add(nodes)
    #index.storage_context.persist()

async def retrieve(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    This handler uses the user input to retrieve memories from the vector db
    """

    print("Inside retrieve")


async def button_tap(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    This handler processes the inline buttons on the menu
    """
    
    global action
    
    data = update.callback_query.data
    text = ''
    markup = None

    if data == STORE_BUTTON:
        # set global status to store
        action = 'store'
        # print prompt to send your message to be stored
        text = STORE_PROMPT
        markup=""
    elif data == RETRIEVE_BUTTON:
        # set global status to retrieve
        action = 'retrieve'
        # print prompt to send your question for retrieval
        text = RETRIEVE_PROMPT
        markup = ""

    # Close the query to end the client-side loading animation
    await update.callback_query.answer()

    # Update message content with corresponding menu section
    await update.callback_query.message.edit_text(
        text,
        ParseMode.HTML,
        reply_markup=markup
    )


def main() -> None:
    application = Application.builder().token(os.getenv('BOT_TOKEN')).build()

    # Register commands
    application.add_handler(CommandHandler("store", store))
    application.add_handler(CommandHandler("retrieve", retrieve))
    application.add_handler(CommandHandler("menu", menu))

    # Register handler for inline buttons
    application.add_handler(CallbackQueryHandler(button_tap))

    # Process messages typed in based on global state
    application.add_handler(MessageHandler(~filters.COMMAND, echo))

    # Start the Bot
    application.run_polling()

if __name__ == '__main__':
    main()
