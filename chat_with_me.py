# *Streamlit are based on code from chat_with_pdf.py 
# *RAG system are based on code from langgraph_chroma_retreiver.ipynb
import streamlit as st
import os
from openai import OpenAI
# *LangChain libary imports
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters.spacy import SpacyTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from IPython.display import Image, display
from langchain_community.document_loaders import PyPDFLoader
import tempfile

client = OpenAI(
	api_key=os.environ["API_KEY"],
	base_url="https://api.ai.it.cornell.edu",
)

st.title("📝 File Q&A with OpenAI") # *title for the Streamlit app interface
uploaded_file = st.file_uploader("Upload an article", type=["txt", "pdf"],accept_multiple_files=True) # *filed uploader that support multiple file types

# *if file is uploaded, user can input a question
question = st.chat_input(
    "Start typing...",
    disabled=not uploaded_file,
)

# *if user didn't enter any message. Will initialize with default message
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask something about the article you have upload"}]

# *display previous message from assistant & user in chat UI (remember history conversation)
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# *Main RAG (if both question & file avaiable)
if question and uploaded_file:

    os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY") # Set OpenAI_API_key to my own variable (so I don't have type twice)
    # *initialize the model
    llm = ChatOpenAI(
        model="openai.gpt-4o-mini",
        temperature=0.2,
    )

    # *Text Splitter method: Recursive Character
    pdf_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 0,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    # *Text Splitter method: SpaCy (sementic/sentence based)
    text_splitter = SpacyTextSplitter(
        chunk_size=180,
        chunk_overlap=20,
        pipeline="en_core_web_sm"
    )

    everything =[]
    # *file loading For Loop
    for file in uploaded_file:
        file_name=file.name
        file_type=file.type

        if file_name.endswith(".pdf"):
            # *Store uploaded PDF temporarily (with assistant with ChatGPT )
            with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as temp_file: # *b/c not support in-memory files. Store them at sys temp directory temporary
                temp_file.write(file.getbuffer())
                temp_path=temp_file.name

            loader = PyPDFLoader(temp_path)
            documents =loader.load() # *will return a list that contain page_content, metadata ->list[Document]

            # annotate metadata for purpose of cite source (with assistant with ChatGPT )
            for i, doc in enumerate(documents):
                doc.metadata["source"] = file_name          # file name
                doc.metadata["page"] = i + 1                # page numbers

            chunk = pdf_splitter.split_documents(documents)

        elif file_name.endswith(".txt"):
            file_content = file.read().decode("utf-8")
            docs = Document(page_content=file_content, metadata={"source": file_name, "page": None})
            chunk = text_splitter.split_documents([docs])

        everything.extend(chunk)

    # *preview chunk to check reliability & correctness
    for i, doc in enumerate(everything):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {doc.metadata['source']}")
        print(doc.page_content)

    embedding=OpenAIEmbeddings(
        model="openai.text-embedding-3-large",
        openai_api_key=os.environ.get("API_KEY"),
        openai_api_base="https://api.ai.it.cornell.edu/v1"
    )

    # *vector DB initalize
    vectorstore = Chroma.from_documents(documents=everything, embedding=embedding)

    
    # template for assistant
    template = """
        You are an assistant for question-answering tasks based on the provide context from retrieved documents to answer the question. 
        If you don't know the answer, respond with: "Sorry... I can't help you with that. I couldn't find that information in the provided documents." 
        Use only the information that is being provided to answer the question. Not other sources should be use.
        Use three sentences maximum and keep the answer concise.

        For each piece of information you provide, cite the source like this:
        - If PDF: "Source: doc_name.pdf (page x)"
        - If TXT: "Source: doc_name.txt"
        
        Question: {question} 
        
        Context: {context} 
        
        Answer:
    """
    prompt = PromptTemplate.from_template(template)

    #3.2 (Retrieval augmented generation pipeline)
    # *retriever initialize
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # *LangGraph State set up
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    # retrieve based on similarity search
    def retrieve(state: State):
        retrieved_docs = vectorstore.similarity_search(state["question"], k=20)
        return {"context": retrieved_docs}

    # generate based on retrived_docs 
    def generate(state: State):
        # docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        # (with assistant with ChatGPT )
        docs_content=""
        for doc in state["context"]:
            source = doc.metadata.get("source","Unknown")
            page = doc.metadata.get("page", None)

            # *Extract filename only
            filename = os.path.basename(source)
            # *dynamic format citation
            if filename.endswith(".pdf"):
                citation = f"{filename} (page {page})"
            elif filename.endswith(".txt"):
                citation = filename
            else:
                citation = "Unknown source"

            docs_content += f"Source: {citation}\n{doc.page_content}\n\n"

        # *Inject into the prompt
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}
    
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    display(Image(graph.get_graph().draw_mermaid_png()))

    #Streamlit continue (chat display)
    # Append the user's question to the messages
    st.session_state.messages.append({"role": "user", "content": question})
    # *display user's question in chat UI
    st.chat_message("user").write(question)

    # *generate response from the assistant using uploaded file and user's question
    with st.chat_message("assistant"):
        result = graph.invoke({"question": question})
        print(f"Context: {result['context']}\n\n")
        response=result['answer']
        print(f"Answer: {result['answer']}")
        st.write(response)

    # Append the assistant's response to the messages
    st.session_state.messages.append({"role": "assistant", "content": response})


