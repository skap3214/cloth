from langchain_community.document_loaders.youtube import YoutubeLoader, _parse_video_id
from langchain.docstore.document import Document
from .config import Config
from typing import List, Optional, Dict

def extract(text: str) -> List[Document]:
    if _parse_video_id(text):
        return _extract_from_youtube(text)
    else:
        return _extract_from_text(text)

def _extract_from_youtube(url: str) -> List[Document]:
    loader = YoutubeLoader.from_youtube_url(url)
    documents = loader.load_and_split(text_splitter=Config.SPLITTER)
    return documents

def _extract_from_text(text: str, metadata: Optional[Dict] = None) -> List[Document]:
    documents = [Document(page_content=text, metadata=metadata)]
    documents = Config.SPLITTER.split_documents(documents)
    return documents