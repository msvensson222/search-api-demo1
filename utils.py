import logging
from typing import Union

from fastapi import HTTPException
from google.cloud import storage
from starlette import status

from variables import GOOGLE_APPLICATION_CREDENTIALS, PROJECT_ID

logging.basicConfig(level=logging.INFO)


def validate_language_code(language_code: str) -> Union[HTTPException, None]:
    # Not yet implemented
    failed = False
    if failed:
        error_msg = f"The input languageCode {language_code} is faulty. It has to be a two-letter " \
                    f"language code in uppercode, ISO-standard."
        logging.exception(error_msg)
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)
    else:
        return None


def download_gcs_file(bucket_name: str, gcs_file_path: str, destination_file_name: str):
    storage_client = storage.Client(project=PROJECT_ID).from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS)
    # Create a bucket object for our bucket
    bucket = storage_client.get_bucket(bucket_name)
    # Create a blob object from the filepath
    blob = bucket.blob(gcs_file_path)
    # Download the file to a destination
    blob.download_to_filename(destination_file_name)
