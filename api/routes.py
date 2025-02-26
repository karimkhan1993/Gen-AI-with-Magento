from fastapi import APIRouter, HTTPException
from utils.text_processing import get_search_tips
from models.schemas import (
    SearchQuery, 
    RecommendationQuery, 
    SearchResponse, 
    RecommendationResponse,
    SyncResponse
)
from services.search import SearchService
from services.recommendations import RecommendationService

router = APIRouter()
search_service = SearchService()
recommendation_service = RecommendationService()

@router.post("/sync", response_model=SyncResponse)
async def sync_data():
    """Endpoint to synchronize product data"""
    try:
        result = search_service.sync_data()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/update_embeddings")
async def update_embeddings():
    """Endpoint to update product embeddings"""
    try:
        result = search_service.update_vector_store()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search", response_model=SearchResponse)
async def search_products(query: SearchQuery):
    """Endpoint for semantic product search"""
    try:
        results = search_service.semantic_search(query.query, query.limit)
        return {
            "results": results,
            "search_tips": get_search_tips()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(query: RecommendationQuery):
    """Endpoint for product recommendations"""
    try:
        recommendations = recommendation_service.get_product_recommendations(
            query.product_id,
            query.limit
        )
        
        return {
            "message": "Recommendations found successfully",
            "source_product_id": query.product_id,
            "recommendations_count": len(recommendations),
            "recommendations": recommendations
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))