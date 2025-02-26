from typing import List, Dict
from utils.logging_config import logger
from services.embedding import EmbeddingService

class RecommendationService:
    """Service for product recommendations"""
    
    def __init__(self):
        """Initialize the recommendation service"""
        self.embedding_service = EmbeddingService()
    
    def get_product_recommendations(self, product_id: int, limit: int = 5) -> List[Dict]:
        """
        Get product recommendations based on similarity
        
        Args:
            product_id: Product ID to get recommendations for
            limit: Maximum number of recommendations to return
        Returns:
            List of recommended products with similarity scores
        """
        try:
            logger.info(f"Getting recommendations for product ID: {product_id}")
            
            # Validate product exists
            if not self.embedding_service.validate_product_exists(product_id):
                raise ValueError(f"Product {product_id} not found")
            
            # Find source product
            source_product = self.embedding_service.find_product_by_id(product_id)
            source_embedding = source_product['embedding']
            source_metadata = source_product['metadata']
            
            # Query for similar products
            results = self.embedding_service.product_collection.query(
                query_embeddings=[source_embedding],
                n_results=limit + 1  # Get extra to account for self-match
            )
            
            # Process recommendations
            recommendations = []
            seen_product_ids = set()
            
            # Zip the metadata and distances together
            for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
                current_id = int(metadata['entity_id'])
                
                # Skip if this is the source product or we've seen it before
                if current_id == product_id or current_id in seen_product_ids:
                    continue
                    
                seen_product_ids.add(current_id)
                
                # Add to recommendations
                recommendations.append({
                    'entity_id': current_id,
                    'sku': metadata['sku'],
                    'name': metadata['name'],
                    'price': float(metadata['price']),
                    'similarity_score': float(distance),
                    'source_product': {
                        'entity_id': product_id,
                        'name': source_metadata['name'],
                        'sku': source_metadata['sku'],
                        'price': float(source_metadata['price'])
                    }
                })
                
                # Stop if we have enough recommendations
                if len(recommendations) >= limit:
                    break
            
            logger.info(f"Found {len(recommendations)} recommendations for product {product_id}")
            return recommendations
            
        except ValueError as e:
            logger.error(f"Value Error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            raise