# Semantic search API
*For fashion in an ecommerce setting, using a hybrid approach to combine both dense vector search with traditional 
keyword-based search.*

## Easy development and deployment with FastAPI + Docker
Containerized API using FastAPI, primary code can be seen in `main.py`. The primary endpoint `/search` takes a query 
and language code and returns the top products for the query. It retrieves products using both an approximate nearest 
neighbor index for dense vector retrieval, as well as Google Retail Search API for simple key-word retrieval.
### Architecture
Below is the workflow for the primary endpoint `/search`, from receiving a query `q` to responding with the most
relevant products.
![](architecture.png)
### Future steps
Add a reranking step, after the hybrid retrieval. Instead of merging the two lists with reciprocal rank fusion, instead
use a reranker model to determine the optimal sorting. For example a cross-encoder or a tree-based model. I currently do
not have any user interactions, so training data is not available. For public ecommerce search relevance data sets, there
exists but none with images (leveraging the multi-modality of this search engine).

