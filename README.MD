Install the required dependencies
pip install sentence-transformers chromadb mysql-connector-python fastapi uvicorn pandas numpy pydantic rapidfuzz

# Magento Gen AI

An AI-powered search and recommendation system for Magento e-commerce platforms.

## Features

- Semantic product search with natural language understanding
- Product recommendations based on content similarity
- Automatic pattern recognition for flower-related queries
- Price filtering and constraints
- Typo-tolerant search with fuzzy matching

## Installation

1. Clone the repository
```bash
git clone https://github.com/karimkhan1993/Gen-AI-with-Magento.git
cd magento-gen-ai
```

2. Install dependencies
```bash
pip install -r requirements.txt

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