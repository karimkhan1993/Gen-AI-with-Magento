Install the required dependencies
pip install sentence-transformers chromadb mysql-connector-python fastapi uvicorn pandas numpy pydantic rapidfuzz

# Setup DB Connection: 
Change magento database connection in magento_gen_ai.py
python magento_gen_ai.py

# Start the service: 
python magento_gen_ai.py

Example API usage:
# Sync for products
curl --location --request POST 'http://localhost:8000/sync'

# Search for products
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "red leather shoes", "limit": 5}'

# Get recommendations
curl -X POST "http://localhost:8000/recommend" \
     -H "Content-Type: application/json" \
     -d '{"product_id": 123, "limit": 5}'