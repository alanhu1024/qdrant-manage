from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

pdf_path = "/Users/alanhu/Library/CloudStorage/OneDrive-个人/创业/01 企业客户/新能源/阳光电源/数据/sungrowpower_data/2022年报.pdf"
loader = DirectoryLoader(pdf_path, glob="*.pdf", loader_cls=PyPDFLoader)
# loader = PyPDFLoader(pdf_path)
docs = loader.load()
print(f"Loaded {len(docs)} documents.")
texts = [doc.page_content for doc in docs]
metadatas = [doc.metadata for doc in docs]

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(data)
# print(f'拆分后现在有{len(texts)}个documents')

embeddings = HuggingFaceEmbeddings(model_name="/Users/alanhu/Model/text2vec-large-chinese/")

coll_name = "YG"
qdrant_client = QdrantClient("8.210.111.212", port=6333)
vector_db = Qdrant(client=qdrant_client, collection_name=coll_name, embeddings=embeddings)

for idx, item in enumerate(metadatas):
    metadatas[idx]['source'] = os.path.basename(item['source'])

vector_db.add_texts(texts=texts, metadatas=metadatas)


