import logging

import numpy as np
import time
from CLIP.clip import clip
import torch
import json
from google.cloud import translate
from utils import download_gcs_file
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.retail import SearchRequest, SearchServiceClient
from google.oauth2 import service_account

import faiss

from data import SearchResponse, Product
from variables import GOOGLE_TRANSLATE_GLOSSARY_ID, PROJECT_ID, \
    GOOGLE_APPLICATION_CREDENTIALS

logging.basicConfig(level=logging.INFO)


class Search:
    def __init__(self, use_fine_tuned_model: bool = False):
        self.use_fine_tuned_model = use_fine_tuned_model
        cred = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
        _ = firebase_admin.initialize_app(cred)
        self.db = firestore.client()
        self.relevant_prod_data = {}
        self.__load()
        self.project_id = PROJECT_ID
        self.retrieval_k = 120
        with open(GOOGLE_APPLICATION_CREDENTIALS) as source:
            info = json.load(source)
        self.creds = service_account.Credentials.from_service_account_info(info)

        # Translations
        if GOOGLE_APPLICATION_CREDENTIALS:
            self.google_translate_client = translate.TranslationServiceClient().from_service_account_json(
                GOOGLE_APPLICATION_CREDENTIALS
            )
        else:
            self.google_translate_client = translate.TranslationServiceClient()
        location = 'us-central1'  # The only supported one
        self.google_translate_glossary_id = GOOGLE_TRANSLATE_GLOSSARY_ID
        self.parent = f"projects/{self.project_id}/locations/{location}"
        glossary = self.google_translate_client.glossary_path(
            self.project_id, location, self.google_translate_glossary_id
        )
        # Ignore case is important!
        self.glossary_config = translate.TranslateTextGlossaryConfig(glossary=glossary, ignore_case=True)

    def query(
            self,
            q: str,
            language_code: str,
            top_n: int = 10
    ) -> SearchResponse:

        if language_code != 'en':
            q = self.translate(q=q, src=language_code)

        start = time.time()
        product_ids_vector_search = self.__query_with_vector_search(q=q)
        print(f"Vector search: Retrieved top-{self.retrieval_k} products in {(time.time() - start):.3f} seconds")

        start = time.time()
        product_ids_keyword_search = self.__query_with_keyword_search(q=q)
        print(f"Keyword search: Retrieved {len(product_ids_keyword_search)} products in {time.time() - start} seconds")

        start = time.time()
        product_ids = self.__rrf(r1=product_ids_vector_search, r2=product_ids_keyword_search)
        print(f"Performed reranking (RRF) in {time.time() - start} seconds")

        product_ids = product_ids[:top_n]

        # Exact match for ID will return first in the response.
        if q in self.product_ids_in_index:
            product_ids = [q] + [product_id for product_id in product_ids if product_id != q]
            if len(product_ids) > top_n:
                product_ids = product_ids[:top_n]

        response = SearchResponse(
            hits=[
                Product(
                    title=self.relevant_prod_data[product_id]["title"],
                    description="",
                    variant_sku=self.relevant_prod_data[product_id]["variant_sku"],
                    handle=self.relevant_prod_data[product_id]["handle"],
                    thumbnail=self.relevant_prod_data[product_id]["thumbnail"]
                )
                for product_id in product_ids
            ]
        )
        return response

    def __query_with_vector_search(self, q: str) -> list[str]:
        text_emb = self.get_text_emb(text=q).astype(np.float32)
        cos_sim, indices = self.index.search(text_emb, self.retrieval_k)
        # print(f"Cosine similarity: {cos_sim[0][:10]}")
        # 1 - 0.5 * (1 - cos_sim)
        # distance = 0.5 * (1 + cos_sim[0])
        return [self.index_mapping[str(index)] for index in indices[0]]

    def __query_with_keyword_search(self, q: str) -> list[str]:
        default_search_placement = (
                "projects/"
                + self.project_id
                + "/locations/global/catalogs/default_catalog/placements/default_search"
        )

        search_request = SearchRequest()
        search_request.placement = default_search_placement  # Placement is used to identify the Serving Config name.
        search_request.query = q
        search_request.visitor_id = "placeholder"  # A unique identifier to track visitors
        search_request.page_size = self.retrieval_k  # Supported maximum is 120

        # Use query expansion to relax the search during multi-word searches
        query_expansion_spec = SearchRequest().QueryExpansionSpec()
        query_expansion_spec.condition = SearchRequest.QueryExpansionSpec.Condition.AUTO
        search_request.query_expansion_spec = query_expansion_spec

        search_response = SearchServiceClient(credentials=self.creds).search(search_request)
        page = next(search_response.pages)  # Just get the first page, containing page_size number of items
        if len(page.results) == 0:
            return []
        else:
            return [result.id for result in page.results]

    @staticmethod
    def __rrf(r1: list[str], r2: list[str]) -> list[str]:
        """
        Performs reciprocal rank fusion for two ranked lists of products, r1 and r2
        """

        def __get_rank(r: list[str], p: str, d: int):
            try:
                return r.index(p) + 1  # 1-indexed instead of 0
            except ValueError:
                return d

        RRF_DEN = 60
        unique_products = set(r1 + r2)
        rrf_scores = []

        for product_id in unique_products:
            default = max((len(r1), len(r2)))
            rank1 = __get_rank(r=r1, p=product_id, d=default)
            rank2 = __get_rank(r=r2, p=product_id, d=default)
            rrf_score = 1 / (RRF_DEN + rank1) + 1 / (RRF_DEN + rank2)
            rrf_scores.append({"product_id": product_id, "rrf_score": rrf_score})

        rrf_scores_sorted = sorted(rrf_scores, key=lambda d: d['rrf_score'], reverse=True)
        return [dct["product_id"] for dct in rrf_scores_sorted]

    def __load_index_mapping(self):
        start = time.time()
        download_gcs_file(
            bucket_name="ms-search-data-private",
            gcs_file_path=f"demo1/index_mapping/index_mapping.json",
            destination_file_name="index_mapping.json"
        )
        with open('index_mapping.json') as f:
            self.index_mapping = json.load(f)
        self.product_ids_in_index = set(self.index_mapping.values())
        logging.info(f"Loaded index mapping in {(time.time() - start):.3f} seconds.")

    def __load_product_catalog(self):
        start = time.time()
        col = self.db.collection('demo1-products')
        docs = col.where("published", "==", True).get()
        for doc in docs:
            d = doc.to_dict()
            self.relevant_prod_data[str(d["productId"])] = {
                "title": d["title"],
                "description": d["description"],
                "variant_sku": [],
                "handle": d["handle"],
                "thumbnail": d["thumbnail"],
                "gender": d["gender"]
            }
        logging.info(
            f"Loaded product info (size: {len(self.relevant_prod_data.keys())}) from firestore "
            f"in {(time.time() - start):.3f} seconds."
        )

    def __load_index(self):
        start = time.time()
        index_prefix = len(self.relevant_prod_data.keys())  # Number of images/products in the index
        if self.use_fine_tuned_model:
            download_gcs_file(
                bucket_name="ms-search-data-private",
                gcs_file_path=f"demo1/index_finetuned/index_{index_prefix}",
                destination_file_name="fine_tuned_index"
            )
            self.index = faiss.read_index("fine_tuned_index")
            logging.info(f"Loaded fine-tuned index in {(time.time() - start):.3f} seconds.")
        else:
            # Download index
            download_gcs_file(
                bucket_name="ms-search-data-private",
                gcs_file_path=f"demo1/index/index_{index_prefix}",
                destination_file_name="index"
            )
            self.index = faiss.read_index("index")
            logging.info(f"Loaded index in {(time.time() - start):.3f} seconds.")

    def __load_model(self):
        if self.use_fine_tuned_model:
            start = time.time()
            download_gcs_file(
                bucket_name="ms-search-data-private",
                gcs_file_path=f"demo1/fine-tuned-model/model_checkpoint/model.pt",
                destination_file_name="fine_tuned_model.pt"
            )
            logging.info(f"Downloaded fine-tuned model in {(time.time() - start):.3f} seconds.")

            start = time.time()
            model_name = "ViT-B/32"
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load(model_name, device=self.device, jit=False)
            checkpoint = torch.load("fine_tuned_model.pt", map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Loaded fine-tuned model in {(time.time() - start):.3f} seconds.")
        else:
            start = time.time()
            model_name = "ViT-B/32"
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load(model_name, device=self.device, jit=True)
            logging.info(f"Loaded model in {(time.time() - start):.3f} seconds.")

    def __load(self):
        self.__load_index_mapping()
        self.__load_product_catalog()
        self.__load_index()
        self.__load_model()

    @torch.no_grad()
    def get_text_emb(self, text: str):
        text_tokens = clip.tokenize(text).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()

    def translate(self, q, src) -> str:
        """Translates a given text using a glossary."""
        start = time.time()
        dest = 'en'  # Always translate to english

        # Supported language codes: https://cloud.google.com/translate/docs/languages
        payload = {
            "contents": [q],
            "target_language_code": dest,
            "source_language_code": src,
            "parent": self.parent,
            "glossary_config": self.glossary_config,
        }
        try:
            response = self.google_translate_client.translate_text(  # TODO: Include await here
                request=payload
            )
        except Exception as e:
            logging.warning(
                f"Could not translate the request. Probably unsupported countrycode. "
                f"Using the original query text instead."
                f"Full error msg: \n{e}"
            )
            return q
        translated_text = response.glossary_translations[0].translated_text
        logging.info(
            f"Translated the query from {src} to english in {(time.time() - start):.3f} seconds."
            f"Result:   {translated_text}"
        )
        return translated_text
