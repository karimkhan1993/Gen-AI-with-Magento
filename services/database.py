import mysql.connector
import pandas as pd
from typing import Dict
from config import get_settings
from utils.logging_config import logger
from utils.text_processing import clean_text

class DatabaseService:
    """Service for database operations"""
    
    def __init__(self, db_config: Dict = None):
        """Initialize the database service"""
        self.db_config = db_config or {
            'host': get_settings().db_host,
            'user': get_settings().db_user,
            'password': get_settings().db_password,
            'database': get_settings().db_name
        }
    
    def connect(self):
        """Create and return a database connection"""
        try:
            return mysql.connector.connect(**self.db_config)
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            raise
    
    def extract_product_data(self) -> pd.DataFrame:
        """Extract product data from Magento database"""
        try:
            conn = self.connect()
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
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess product data for embedding"""
        df['name'] = df['name'].fillna('')
        df['description'] = df['description'].fillna('')
        df['clean_text'] = (
            df['name'].apply(clean_text) + ' ' + 
            df['description'].apply(clean_text)
        )
        return df