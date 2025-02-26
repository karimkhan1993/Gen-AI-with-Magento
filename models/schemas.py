from pydantic import BaseModel
from typing import List, Dict, Optional, Union, Any

class SearchQuery(BaseModel):
    query: str
    limit: int = 5

class RecommendationQuery(BaseModel):
    product_id: int
    limit: int = 5

class ProductMetadata(BaseModel):
    entity_id: int
    sku: str
    name: str
    price: float

class SearchResult(BaseModel):
    entity_id: int
    sku: str
    name: str
    price: float
    relevance_score: float
    matched_terms: Dict[str, str]

class SourceProduct(BaseModel):
    entity_id: int
    name: str
    sku: str
    price: float

class Recommendation(BaseModel):
    entity_id: int
    sku: str
    name: str
    price: float
    similarity_score: float
    source_product: SourceProduct

class SearchResponse(BaseModel):
    results: List[SearchResult]
    search_tips: Dict[str, Any]

class RecommendationResponse(BaseModel):
    message: str
    source_product_id: int
    recommendations_count: int
    recommendations: List[Recommendation]

class SyncResponse(BaseModel):
    message: str
    product_count: int
