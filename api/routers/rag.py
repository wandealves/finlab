from config.settings import settings
from fastapi import APIRouter
from models.rag import RagRequest, RagResponse
from services.rag import RagService
from services.search import SearchService

router = APIRouter()

search_service = SearchService(
    qdrant_url=settings.qdrant_url,
    qdrant_api_key=settings.qdrant_api_key,
    collection_name=settings.collection_name,
)

rag_service = RagService(search_service=search_service)


@router.post("/rag", response_model=RagResponse)
def rag(request: RagRequest):
    return rag_service.generate_answer(request.query, request.limit)
