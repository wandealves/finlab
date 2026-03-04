from config.settings import settings
from fastapi import FastAPI
from models.rag import RagRequest, RagResponse
from models.search import SearchRequest, SearchResponse
from services.rag import RagService
from services.search import SearchService

app = FastAPI(title="Financial Search API")

search_service = SearchService(
    qdrant_url=settings.qdrant_url,
    qdrant_api_key=settings.qdrant_api_key,
    collection_name=settings.collection_name,
)

rag_service = RagService(search_service=search_service)


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    return search_service.search(request.query, request.limit)


@app.post("/rag", response_model=RagResponse)
def rag(request: RagRequest):
    return rag_service.generate_answer(request.query, request.limit)


@app.get("/")
def root():
    return {"status": "online"}
