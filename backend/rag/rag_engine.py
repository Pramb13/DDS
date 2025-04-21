import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

pinecone.init(api_key="pcsk_4m9ssG_9YGimgnHkz8z6r766SBvTXUXNz5RxB8MN9ttGRQLV9vXj38FxJCRkod14mn2RCB", environment="YOUR_ENV")
index = pinecone.Index("driver-logs")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Pinecone(index, embeddings.embed_query, "text")
llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.1", model_kwargs={"temperature": 0.6, "max_new_tokens": 512})
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

def query_logs(question):
    return qa.run(question)
