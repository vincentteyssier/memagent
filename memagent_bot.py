import logging, os, sys
from dotenv import load_dotenv

from telegram import Update, ForceReply, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, ContextTypes, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler
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
from local_embeddings import LocalHuggingFaceEmbeddings

from llama_index import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.faiss import FaissVectorStore

# initialize the vector store
# dimensions of text-ada-embedding-002
d = 1536
faiss_index = faiss.IndexFlatL2(d)
vector_store = FaissVectorStore(faiss_index=faiss_index)

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





load_dotenv()

# Store bot screaming status
screaming = False

# Pre-assign menu text
FIRST_MENU = "<b>Menu</b>\n\nChoose your action:"

# Pre-assign button text
STORE_BUTTON = "Store Note"
RETRIEVE_BUTTON = "Retrieve Memory"
TUTORIAL_BUTTON = "Tutorial"

# Build keyboards
FIRST_MENU_MARKUP = InlineKeyboardMarkup([[
    InlineKeyboardButton(STORE_BUTTON, callback_data=STORE_BUTTON),
    [InlineKeyboardButton(RETRIEVE_BUTTON, callback_data=RETRIEVE_BUTTON)]
]])



async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    This function would be added to the application as a handler for messages coming from the Bot API
    """

    # Print to console
    print(f'{update.message.from_user.first_name} wrote {update.message.text}')

    if screaming and update.message.text:
        await context.bot.send_message(
            update.message.chat_id,
            update.message.text.upper(),
            # To preserve the markdown, we attach entities (bold, italic...)
            entities=update.message.entities
        )
    else:
        # This is equivalent to forwarding, without the sender's name
        await update.message.copy(update.message.chat_id)


async def scream(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    This function handles the /scream command
    """

    global screaming
    screaming = True


async def whisper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    This function handles /whisper command
    """

    global screaming
    screaming = False


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


async def button_tap(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    This handler processes the inline buttons on the menu
    """

    data = update.callback_query.data
    text = ''
    markup = None

    if data == STORE_BUTTON:
        text = FIRST_MENU
        markup = FIRST_MENU_MARKUP
    elif data == RETRIEVE_BUTTON:
        text = FIRST_MENU
        markup = FIRST_MENU_MARKUP

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
    application.add_handler(CommandHandler("scream", scream))
    application.add_handler(CommandHandler("whisper", whisper))
    application.add_handler(CommandHandler("menu", menu))

    # Register handler for inline buttons
    application.add_handler(CallbackQueryHandler(button_tap))

    # Echo any message that is not a command
    application.add_handler(MessageHandler(~filters.COMMAND, echo))

    # Start the Bot
    application.run_polling()

if __name__ == '__main__':
    main()
