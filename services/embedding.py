import pandas as pd
import numpy as np
import chromadb
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from config import get_settings
from utils.logging_config import logger

class EmbeddingService:
    """Service for generating and managing embeddings"""
    
    def __init__(self, model_name: str = None):
        """Initialize the embedding service"""
        self.model_name = model_name or get_settings().embedding_model
        self.model = SentenceTransformer(self.model_name)
        
        # Use a persistent path for ChromaDB
        self.chroma_client = chromadb.Client(chromadb.Settings(
            persist_directory="./chroma_db"
        ))
        
        try:
            self.product_collection = self.chroma_client.get_collection(name="magento_products")
            logger.info("Retrieved existing collection 'magento_products'")
        except:
            self.product_collection = self.chroma_client.create_collection(
                name="magento_products",
                metadata={"description": "Magento product embeddings"}
            )
            logger.info("Created new collection 'magento_products'")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        return self.model.encode(texts)
    
    def find_product_by_id(self, product_id: int) -> Dict:
        """
        Find a product by its ID in the collection
        
        Args:
            product_id: The product ID to search for
        Returns:
            Dict containing product data if found
        """
        try:
            # Convert ID to string for ChromaDB
            str_id = str(product_id)
            results = self.product_collection.get(
                ids=[str_id],
                include=['embeddings', 'metadatas']
            )
            
            # Check if we got results
            if not results['ids'] or len(results['ids']) == 0:
                raise ValueError(f"Product {product_id} not found")
                
            return {
                'embedding': results['embeddings'][0],
                'metadata': results['metadatas'][0]
            }
        except Exception as e:
            logger.error(f"Error finding product {product_id}: {str(e)}")
            raise
    
    def validate_product_exists(self, product_id: int) -> bool:
        """
        Validate if a product exists in the collection
        
        Args:
            product_id: Product ID to validate
        Returns:
            bool: True if product exists, False otherwise
        """
        try:
            # Only use string format for ChromaDB
            str_id = str(product_id)
            
            results = self.product_collection.get(
            ids=[str_id],
            include=['metadatas']
            )
        
            # Check if results contain any data
            return bool(results and results['metadatas'] and len(results['metadatas']) > 0)
        except Exception as e:
            logger.error(f"Error validating product: {str(e)}")
            return False
    
    def reset_collection(self):
        """Reset the product collection"""
        try:
            self.chroma_client.delete_collection("magento_products")
            self.product_collection = self.chroma_client.create_collection(
                name="magento_products",
                metadata={"description": "Magento product embeddings"}
            )
            logger.info("Collection reset successfully")
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            raise
    
    def update_collection(self, df, embeddings):
        """Update the product collection with new data and embeddings"""
        try:
            self.product_collection.upsert(
                ids=[str(i) for i in df['entity_id']],
                embeddings=embeddings.tolist(),
                metadatas=[{
                    'entity_id': int(row['entity_id']),
                    'sku': row['sku'],
                    'name': row['name'],
                    'price': float(row['price']) if pd.notnull(row['price']) else 0.0
                } for _, row in df.iterrows()],
                documents=df['clean_text'].tolist()
            )
            logger.info(f"Updated collection with {len(df)} products")
        except Exception as e:
            logger.error(f"Error updating collection: {str(e)}")
            raise