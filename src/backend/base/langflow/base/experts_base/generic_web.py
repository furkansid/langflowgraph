
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma

class WebLoader():

    def __init__(self, urls, ollama_embedder_model='mxbai-embed-large') -> None:
        self.urls = urls
        self.ollama_embedder_model = ollama_embedder_model



    async def read_web(self, url):
        pages = []
        async for page in WebBaseLoader(url).alazy_load():
            pages.append(page)
        return pages
    
    async def read_webs(self):
        pages = []
        for url in self.urls:
            pages += await self.read_web(url)
        return pages

    
    async def retreiver(self, num_of_docs=3):
        embeddings = OllamaEmbeddings(model=self.ollama_embedder_model)
        pages = await self.read_webs()
        # vector = await InMemoryVectorStore.afrom_documents(pages, embeddings)
        doc_splits = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=20).split_documents(pages)
        vectorstore = await Chroma.afrom_documents(documents=doc_splits, collection_name='web-Oct-S1', embedding=embeddings)
        retriever = vectorstore.as_retriever(k=num_of_docs)
        return retriever
    
    async def ask(self, question, num_of_answers=3):
        retriever = await self.retreiver(num_of_docs=num_of_answers)
        docs = await retriever.ainvoke(question)
        return docs