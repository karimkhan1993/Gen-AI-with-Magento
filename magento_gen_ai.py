import mysql.connector
from sentence_transformers import SentenceTransformer
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Dict
import re
from datetime import datetime
import json
import logging
from rapidfuzz import process

class FlowerSearchPatterns:
    """Patterns for common flower search queries"""
    
    OCCASIONS = {
        'anniversary': ['anniversary', 'wedding anniversary', 'yearly celebration'],
        'birthday': ['birthday', 'bday', 'birth day celebration'],
        'wedding': ['wedding', 'bridal', 'marriage'],
        'sympathy': ['sympathy', 'condolence', 'funeral', 'grief'],
        'romance': ['romance', 'romantic', 'love', 'valentine', "valentine's day"],
        'congratulations': ['congratulations', 'congrats', 'graduation', 'promotion'],
        'get well': ['get well', 'recovery', 'hospital', 'feel better']
    }
    
    COLORS = {
        'red': ['red', 'crimson', 'scarlet'],
        'pink': ['pink', 'rose pink', 'blush'],
        'white': ['white', 'pure white', 'snow white'],
        'yellow': ['yellow', 'golden', 'sunny'],
        'purple': ['purple', 'lavender', 'violet'],
        'orange': ['orange', 'coral', 'peach'],
        'blue': ['blue', 'azure', 'sky blue'],
        'mixed': ['mixed', 'rainbow', 'multicolor', 'colorful']
    }
    
    FLOWER_TYPES = {
        'roses': ['rose', 'roses', 'red rose', 'garden rose'],
        'lilies': ['lily', 'lilies', 'asiatic lily'],
        'carnations': ['carnation', 'carnations'],
        'chrysanthemums': ['chrysanthemum', 'mums', 'chrysanths'],
        'orchids': ['orchid', 'orchids', 'phalaenopsis'],
        'sunflowers': ['sunflower', 'sunflowers', 'sun flower'],
        'tulips': ['tulip', 'tulips', 'dutch tulip'],
        'mixed flowers': ['mixed flowers', 'mixed bouquet', 'flower mix']
    }
    
    SIZES = {
        'small': ['small', 'petit', 'tiny', 'compact'],
        'medium': ['medium', 'standard', 'regular'],
        'large': ['large', 'big', 'grand', 'deluxe'],
        'extra large': ['extra large', 'xl', 'luxury', 'premium']
    }
    
    PRICE_RANGES = {
        'budget': ['cheap', 'affordable', 'budget', 'under $50', 'inexpensive'],
        'medium': ['moderate', 'mid-range', '$50-$100'],
        'premium': ['premium', 'luxury', 'expensive', 'over $100']
    }
    
    ARRANGEMENTS = {
        'bouquet': ['bouquet', 'bunch', 'hand-tied'],
        'vase': ['vase arrangement', 'in vase', 'with vase'],
        'basket': ['basket', 'flower basket', 'gift basket'],
        'box': ['box', 'flower box', 'hat box']
    }

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MagentoGenAI:
    def __init__(self, db_config: Dict):
        """Initialize the Magento Gen AI system"""
        self.db_config = db_config
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.Client()
        self.patterns = FlowerSearchPatterns()
        try:
            self.product_collection = self.chroma_client.get_collection(name="magento_products")
            logger.info("Retrieved existing collection 'magento_products'")
        except:
            self.product_collection = self.chroma_client.create_collection(
                name="magento_products",
                metadata={"description": "Magento product embeddings"}
            )
            logger.info("Created new collection 'magento_products'")

    def find_closest_match(self, query: str, options: List[str], threshold: int = 80) -> str:
        """
        Find the closest match for a query from a list of options using rapidfuzz.
        
        Args:
            query: The query string to match.
            options: List of possible options to match against.
            threshold: Minimum similarity score to consider a match.
        Returns:
            The closest match if above threshold, otherwise None.
        """
        result = process.extractOne(query, options, score_cutoff=threshold)
        if result:
            match, score, _ = result  # Unpack the result if it's not None
            return match
        return None  # Return None if no match is found

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
            # Try to get the product using different ID formats
            id_variations = [
                str(product_id),
                f"{product_id}",
                product_id
            ]
            
            for id_var in id_variations:
                try:
                    results = self.product_collection.get(
                        ids=[id_var],
                        include=['metadatas']
                    )
                    if results and results['metadatas']:
                        return True
                except:
                    continue
            
            return False
        except Exception as e:
            print(f"Error validating product: {str(e)}")
            return False

    def parse_search_query(self, query: str) -> Dict[str, str]:
        """
        Parse the search query to identify key search components with typo tolerance.
        
        Args:
            query: Raw search query string
        Returns:
            Dict containing identified search components
        """
        query = query.lower()
        components = {
            'occasion': None,
            'color': None,
            'flower_type': None,
            'size': None,
            'price_range': None,
            'arrangement': None,
            'price_limit': None,
            'cleaned_query': query
        }
        
        # Extract price limit first
        cleaned_query, price_limit, _ = self.extract_price_constraint(query)
        components['price_limit'] = price_limit
        components['cleaned_query'] = cleaned_query
        
        # Match patterns with typo tolerance using rapidfuzz
        for occasion, patterns in self.patterns.OCCASIONS.items():
            match = self.find_closest_match(cleaned_query, patterns)
            if match:
                components['occasion'] = occasion
                
        for color, patterns in self.patterns.COLORS.items():
            match = self.find_closest_match(cleaned_query, patterns)
            if match:
                components['color'] = color
                
        for flower, patterns in self.patterns.FLOWER_TYPES.items():
            match = self.find_closest_match(cleaned_query, patterns)
            if match:
                components['flower_type'] = flower
                
        for size, patterns in self.patterns.SIZES.items():
            match = self.find_closest_match(cleaned_query, patterns)
            if match:
                components['size'] = size
                
        for price_range, patterns in self.patterns.PRICE_RANGES.items():
            match = self.find_closest_match(cleaned_query, patterns)
            if match:
                components['price_range'] = price_range
                
        for arrangement, patterns in self.patterns.ARRANGEMENTS.items():
            match = self.find_closest_match(cleaned_query, patterns)
            if match:
                components['arrangement'] = arrangement
        
        return components
    
    def enhance_search_query(self, components: Dict[str, str]) -> str:
        """
        Create an enhanced search query based on identified components
        
        Args:
            components: Dictionary of search components
        Returns:
            Enhanced search query string
        """
        query_parts = []
        
        if components['occasion']:
            query_parts.append(f"perfect for {components['occasion']}")
            
        if components['flower_type']:
            query_parts.append(components['flower_type'])
            
        if components['color']:
            query_parts.append(f"{components['color']} colored")
            
        if components['size']:
            query_parts.append(f"{components['size']} size")
            
        if components['arrangement']:
            query_parts.append(f"in {components['arrangement']}")
            
        # Add original cleaned query if it contains unique terms
        original_terms = set(components['cleaned_query'].split())
        enhanced_terms = set(' '.join(query_parts).split())
        unique_terms = original_terms - enhanced_terms
        if unique_terms:
            query_parts.append(' '.join(unique_terms))
            
        return ' '.join(query_parts)
        
    def connect_db(self):
        """Create database connection"""
        return mysql.connector.connect(**self.db_config)

    def clean_text(self, text: str) -> str:
        """Clean product text by removing HTML and special characters"""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        return ' '.join(text.split())

    def extract_product_data(self) -> pd.DataFrame:
        """Extract product data from Magento database"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            query = """
                SELECT 
                    cpe.entity_id,
                    cpe.sku,
                    cpev_name.value as name,
                    cpev_desc.value as description,
                    cpep.value as price
                FROM catalog_product_entity cpe
                LEFT JOIN catalog_product_entity_varchar cpev_name 
                    ON cpe.entity_id = cpev_name.entity_id 
                    AND cpev_name.attribute_id = (
                        SELECT attribute_id 
                        FROM eav_attribute 
                        WHERE attribute_code = 'name' 
                        AND entity_type_id = 4
                    )
                LEFT JOIN catalog_product_entity_text cpev_desc 
                    ON cpe.entity_id = cpev_desc.entity_id 
                    AND cpev_desc.attribute_id = (
                        SELECT attribute_id 
                        FROM eav_attribute 
                        WHERE attribute_code = 'description' 
                        AND entity_type_id = 4
                    )
                LEFT JOIN catalog_product_entity_decimal cpep 
                    ON cpe.entity_id = cpep.entity_id 
                    AND cpep.attribute_id = (
                        SELECT attribute_id 
                        FROM eav_attribute 
                        WHERE attribute_code = 'price' 
                        AND entity_type_id = 4
                    )
                WHERE cpe.type_id = 'simple'
            """
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            logger.info(f"Extracted {len(df)} products from database")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting product data: {str(e)}")
            raise
    
    def sync_data(self):
        """Synchronize product data"""
        try:
            logger.info("Starting data synchronization")
            
            # Get fresh data
            df = self.extract_product_data()
            df = self.preprocess_data(df)
            
            # Generate embeddings
            embeddings = self.model.encode(df['clean_text'].tolist())
            
            # Recreate collection
            self.chroma_client.delete_collection("magento_products")
            self.product_collection = self.chroma_client.create_collection(
                name="magento_products",
                metadata={"description": "Magento product embeddings"}
            )
            
            # Add all products
            self.product_collection.add(
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
            
            product_count = len(df)
            logger.info(f"Synchronized {product_count} products successfully")
            
            return {
                "message": "Data synchronized successfully",
                "product_count": product_count
            }
            
        except Exception as e:
            logger.error(f"Error during sync: {str(e)}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess product data for embedding"""
        df['name'] = df['name'].fillna('')
        df['description'] = df['description'].fillna('')
        df['clean_text'] = (
            df['name'].apply(self.clean_text) + ' ' + 
            df['description'].apply(self.clean_text)
        )
        return df

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for product texts"""
        return self.model.encode(texts)

    def update_vector_store(self):
        """Update vector store with latest product data"""
        try:
            # Extract and preprocess data
            df = self.extract_product_data()
            df = self.preprocess_data(df)
            
            # Generate embeddings
            embeddings = self.generate_embeddings(df['clean_text'].tolist())
            
            # Update collection
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
            
        except Exception as e:
            raise Exception(f"Error updating vector store: {str(e)}")

    def extract_price_constraint(self, query: str) -> tuple:
        """
        Extract price constraints from the query
        
        Args:
            query: Search query string
        Returns:
            tuple: (cleaned_query, price_limit, comparison_type)
        """
        # Price patterns
        price_patterns = [
            r'under\s*\$?\s*(\d+)',
            r'less than\s*\$?\s*(\d+)',
            r'below\s*\$?\s*(\d+)',
            r'\$?\s*(\d+)\s*or less',
            r'cheaper than\s*\$?\s*(\d+)',
        ]
        
        cleaned_query = query
        price_limit = None
        comparison_type = 'less'  # Could be 'less' or 'more'
        
        for pattern in price_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                price_limit = float(match.group(1))
                # Remove the price constraint from the query
                cleaned_query = re.sub(pattern, '', query, flags=re.IGNORECASE).strip()
                break
                
        return cleaned_query, price_limit, comparison_type

    def semantic_search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Perform semantic search with enhanced query understanding
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
        """
        # Parse and enhance the query
        components = self.parse_search_query(query)
        enhanced_query = self.enhance_search_query(components)
        
        # Get semantic search results
        query_embedding = self.model.encode(enhanced_query)
        search_limit = limit * 3 if components['price_limit'] is not None else limit
        
        results = self.product_collection.query(
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
            
            # Find source product
            source_product = self.find_product_by_id(product_id)
            source_embedding = source_product['embedding']
            source_metadata = source_product['metadata']
            
            # Query for similar products
            results = self.product_collection.query(
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

# FastAPI application
app = FastAPI()

class SearchQuery(BaseModel):
    query: str
    limit: int = 5

class RecommendationQuery(BaseModel):
    product_id: int
    limit: int = 5

# Initialize Magento Gen AI with database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'dev_drinkindia_db'
}

magento_ai = MagentoGenAI(db_config)

@app.post("/sync")
async def sync_data():
    """Endpoint to synchronize product data"""
    try:
        result = magento_ai.sync_data()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/update_embeddings")
async def update_embeddings():
    """Endpoint to update product embeddings"""
    try:
        magento_ai.update_vector_store()
        return {"message": "Product embeddings updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_products(query: SearchQuery):
    """Endpoint for semantic product search"""
    try:
        results = magento_ai.semantic_search(query.query, query.limit)
        return {
            "results": results,
            "search_tips": get_search_tips()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/recommend")
async def get_recommendations(query: RecommendationQuery):
    """Endpoint for product recommendations"""
    try:
        recommendations = magento_ai.get_product_recommendations(
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

def get_search_tips() -> Dict[str, List[str]]:
    """Return helpful search tips for users"""
    return {
        "example_searches": [
            "red roses with vase for anniversary under $75",
            "small pink birthday bouquet",
            "luxury mixed flowers for wedding",
            "sympathy lilies arrangement",
            "affordable sunflower basket"
        ],
        "search_components": {
            "occasions": list(FlowerSearchPatterns.OCCASIONS.keys()),
            "colors": list(FlowerSearchPatterns.COLORS.keys()),
            "flower_types": list(FlowerSearchPatterns.FLOWER_TYPES.keys()),
            "sizes": list(FlowerSearchPatterns.SIZES.keys()),
            "arrangements": list(FlowerSearchPatterns.ARRANGEMENTS.keys()),
            "price_ranges": list(FlowerSearchPatterns.PRICE_RANGES.keys())
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)