import os
import json
import traceback
import boto3
from loguru import logger

ENV = os.environ.get("ENV", "dev")


def load() -> dict:
    try:
        logger.info("Loading Secrets")
        secret_name = "sandx.ai"
        region_name = "us-east-1"
        session = boto3.session.Session(  # type: ignore
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=region_name,
        )
        client = session.client(
            service_name="secretsmanager",
            region_name=region_name,
        )

        get_secret_value_response = client.get_secret_value(SecretId=secret_name)

        secret_string = get_secret_value_response["SecretString"]
        secrets = json.loads(secret_string)

        for key, value in secrets.items():
            os.environ[key] = value

        if ENV == "prod":
            os.environ["DATABASE_URL"] = secrets["PROD_DATABASE_URL"]
        else:
            os.environ["DATABASE_URL"] = secrets["DEV_DATABASE_URL"]

        return secrets
    except Exception as e:
        logger.error(f"Failed to load secrets: {e}. {traceback.format_exc()}")
        exit(1)


load()
