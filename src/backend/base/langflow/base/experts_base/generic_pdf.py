
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma

class PDFLoader():

    def __init__(self, paths, ollama_embedder_model='mxbai-embed-large') -> None:
        self.paths = paths
        self.ollama_embedder_model = ollama_embedder_model


    async def read_pdf(self, path):
        loader = PyPDFLoader(path)
        pages = []
        async for page in loader.alazy_load():
            pages.append(page)
        return pages
    
    async def read_pdfs(self):
        pages = []
        for path in self.paths:
            pages += await self.read_pdf(path)
        return pages

    
    async def retreiver(self, num_of_docs=3):
        embeddings = OllamaEmbeddings(model=self.ollama_embedder_model)
        pages = await self.read_pdfs()
        # vector = await InMemoryVectorStore.afrom_documents(pages, embeddings)
        doc_splits = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=20).split_documents(pages)
        vectorstore = await Chroma.afrom_documents(documents=doc_splits, collection_name='pdf-Oct-S1', embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k":num_of_docs})
        return retriever
    
    async def ask(self, question, num_of_answers=3):
        retriever = await self.retreiver(num_of_docs=num_of_answers)
        docs = await retriever.ainvoke(question)
        return docs