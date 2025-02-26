from typing import List, Dict
import pandas as pd
from config import get_settings
from utils.logging_config import logger
from utils.text_processing import parse_search_query, enhance_search_query
from services.embedding import EmbeddingService
from services.database import DatabaseService

class SearchService:
    """Service for semantic search operations"""
    
    def __init__(self):
        """Initialize the search service"""
        self.embedding_service = EmbeddingService()
        self.db_service = DatabaseService()
        self.match_threshold = get_settings().match_threshold
    
    def semantic_search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Perform semantic search with enhanced query understanding
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
        """
        # Parse and enhance the query
        components = parse_search_query(query, self.match_threshold)
        enhanced_query = enhance_search_query(components)
        
        # Get semantic search results
        query_embedding = self.embedding_service.model.encode(enhanced_query)
        search_limit = limit * 3 if components['price_limit'] is not None else limit
        
        results = self.embedding_service.product_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=search_limit
        )
        
        # Process and filter results
        filtered_results = []
        for meta, score in zip(results['metadatas'][0], results['distances'][0]):
            price = float(meta['price'])
            
            # Apply price filter if specified
            if components['price_limit'] is not None and price > components['price_limit']:
                continue
                
            filtered_results.append({
                'entity_id': meta['entity_id'],
                'sku': meta['sku'],
                'name': meta['name'],
                'price': price,
                'relevance_score': score,
                'matched_terms': {
                    k: v for k, v in components.items() 
                    if v is not None and k != 'cleaned_query'
                }
            })
            
            if len(filtered_results) >= limit:
                break
        
        return filtered_results
    
    def sync_data(self):
        """Synchronize product data"""
        try:
            logger.info("Starting data synchronization")
            
            # Get fresh data
            df = self.db_service.extract_product_data()
            df = self.db_service.preprocess_data(df)
            
            # Generate embeddings
            embeddings = self.embedding_service.generate_embeddings(df['clean_text'].tolist())
            
            # Reset and recreate collection
            #self.embedding_service.reset_collection()
            
            # Add all products
            self.embedding_service.update_collection(df, embeddings)
            
            product_count = len(df)
            logger.info(f"Synchronized {product_count} products successfully")
            
            return {
                "message": "Data synchronized successfully",
                "product_count": product_count
            }
            
        except Exception as e:
            logger.error(f"Error during sync: {str(e)}")
            raise
    
    def update_vector_store(self):
        """Update vector store with latest product data"""
        try:
            # Extract and preprocess data
            df = self.db_service.extract_product_data()
            df = self.db_service.preprocess_data(df)
            
            # Generate embeddings
            embeddings = self.embedding_service.generate_embeddings(df['clean_text'].tolist())
            
            # Update collection
            self.embedding_service.update_collection(df, embeddings)
            
            return {"message": "Product embeddings updated successfully"}
            
        except Exception as e:
            logger.error(f"Error updating vector store: {str(e)}")
            raise