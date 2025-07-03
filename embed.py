import os

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import UpstageDocumentParseLoader
from langchain_upstage import UpstageEmbeddings
from pinecone import Pinecone, ServerlessSpec
import re
from glob import glob
from langchain.docstore.document import Document

load_dotenv()

def remove_html_tags(text):
    # <로 시작하고 영어 알파벳 또는 /로 시작하는 태그만 제거
    clean_text = re.sub(r'<[a-zA-Z/][^>]*>', '', text)
    return clean_text

def split_by_article(text):
    # 1. 제일 첫 조항 위치 찾기
    first_article_match = re.search(r'제\d+조\s*\(.*?\)', text)
    if not first_article_match:
        return None, []

    first_index = first_article_match.start()

    # 2. 머리말 따로 저장
    metadata = text[:first_index].strip()

    # '제숫자조'로 시작하고, 다음 '제숫자조'가 나오기 전까지를 하나의 단락으로 분리
    pattern = r'(제\d+조\s*\(.*?\)[\s\S]*?)(?=제\d+조\s*\(.*?\)|\Z)'
    matches = re.findall(pattern, text)
    return metadata, matches

# upstage
embedding_upstage = UpstageEmbeddings(model='solar-embedding-1-large')

pinecone_api_key = os.environ.get('PINECONE_API_KEY')
pc = Pinecone(api_key=pinecone_api_key)
index_name = 'house-lease'
folder_path = '.\임대차법률문서'
pdf_paths = glob(os.path.join(folder_path, '*.pdf'))

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=4096,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

document_parse_loader = UpstageDocumentParseLoader(
    pdf_paths,
    output_format='html',  # 결과물 형태 : HTML
    coordinates=False)  # 이미지 OCR 좌표계 가지고 오지 않기

docs = document_parse_loader.load()

documents = []
for doc in docs:
  metadata, articles = split_by_article(remove_html_tags(doc.page_content))
  for article in articles:
    documents.append(
        Document(
            page_content=article,
            metadata={'raw_metadata': metadata}
        )
    )

PineconeVectorStore.from_documents(
   documents, embedding_upstage, index_name=index_name
)
print('end')