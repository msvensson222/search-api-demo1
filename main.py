import logging

import uvicorn
from fastapi import FastAPI, Body, Security, HTTPException
from fastapi.responses import Response
from fastapi.security.api_key import APIKeyHeader
from starlette import status

from Search import Search
from data import SearchResponse
from variables import USE_FINE_TUNED_INDEX, API_KEY

APIDocumentationTags = [
    {
        "name": "Hybrid Search",
        "description": "Hybrid Search API, providing a semantic search in an ecommerce settting."
    },
]
logging.basicConfig(level=logging.INFO)


class API(FastAPI):
    def __init__(self, openapi_tags):
        super().__init__(openapi_tags=openapi_tags)
        self.build()
        self.search_helper = Search(use_fine_tuned_model=USE_FINE_TUNED_INDEX)

    def build(self):
        api_key_header_auth = APIKeyHeader(name="apiKey", auto_error=True)

        def get_api_key(api_key_header: str = Security(api_key_header_auth)):
            if api_key_header != API_KEY:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API Key",
                )

        @self.post("/search",
                   tags=["Search for products"],
                   description="Given a query, return a set of hits",
                   name="",
                   response_model=SearchResponse,
                   dependencies=[Security(get_api_key)]
                   )
        async def search(q: str = Body(...,
                                       example="a shirt",
                                       description="Product filter identifier"
                                       ),
                         languageCode: str = Body(...,
                                                  example="en",
                                                  description="Two letter language code, lowercase"
                                                  ),
                         topN: int = Body(default=10,
                                          example=10,
                                          description="Number of records to return"
                                          )
                         ):

            logging.info(f" -- Query: {q} -- Language code: {languageCode} -- topN: {topN}")

            response = self.search_helper.query(
                q=q,
                language_code=languageCode,
                top_n=topN
            )
            return response

        @self.get("/",
                  name="",
                  response_class=Response,
                  dependencies=[Security(get_api_key)])
        async def healthCheck():
            return Response(status_code=200)


app = API(openapi_tags=APIDocumentationTags)

if __name__ == "__main__":
    # uvicorn main:app --reload --host 0.0.0.0 --port 5000
    uvicorn.run("main:app", reload=False, port=5000)
