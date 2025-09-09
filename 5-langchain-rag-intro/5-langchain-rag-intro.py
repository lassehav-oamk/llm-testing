import getpass
import os

if not os.environ.get('GOOGLE_API_KEY'):
    os.environ['GOOGLE_API_KEY'] = getpass.getpass("Enter your Google API Key for Gemini Access: ")

# init_chat_model init_chat_model function to easily initialize various chat models from different providers like OpenAI, Anthropic, Google, and more. This method simplifies the setup process by handling imports and configurations.
from langchain.chat_models import init_chat_model
llm = init_chat_model('gemini-2.5-flash', model_provider='google_genai')

from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)



import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict


# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
all_splits = text_splitter.split_documents(docs)
print(f"Split blog post into {len(all_splits)} sub-documents.")

# Do the indexing of the chunks
vector_store.add_documents(all_splits)

# Next define the prompt by loading it from the Langchain prompt hub
# Here is the direct link 
# https://smith.langchain.com/hub/rlm/rag-prompt?_gl=1*1tv8aoc*_ga*Nzg2NjEzOTg3LjE3NTcwNjUyMTk.*_ga_47WX3HKKY2*czE3NTcwNjg0OTIkbzIkZzEkdDE3NTcwNjg0OTkkajUzJGwwJGgw
prompt = hub.pull("rlm/rag-prompt")

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps for the langraph
# Each step is a function that takes the current state and returns an updated state.
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Compile application and test
# Langraph provides a StateGraph builder to easily define the application flow
# application flow is defined as a graph of states and transitions.
# Langgraphs uses a state machine approach to manage the flow of data and operations. Thats what the class State is for.
# Here we have a simple linear flow: START -> retrieve -> generate
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Now we can run the application by invoking the graph with an initial state.
# Invoke will return the final state once the execution is complete.
response = graph.invoke({"question": "What is Task Decomposition?"})
print(response["answer"])
    
# another option is to use streaming response
for step in graph.stream({"question": "What is Task Decomposition?"}, stream_mode="messages"):
    print(f"Step: {step[0]}, Output: {step[1]}")

