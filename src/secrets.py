import os
import json
import boto3

ENV = os.environ.get("ENV", "dev")


def load() -> dict:
    secret_name = "sandx.ai"
    region_name = "us-east-1"
    session = boto3.session.Session()  # type: ignore
    client = session.client(
        service_name="secretsmanager",
        region_name=region_name,
    )

    get_secret_value_response = client.get_secret_value(SecretId=secret_name)

    secret_string = get_secret_value_response["SecretString"]
    secrets = json.loads(secret_string)

    for key, value in secrets.items():
        os.environ[key] = value

    if ENV == "dev":
        os.environ["DATABASE_URL"] = secrets["DEV_DATABASE_URL"]
    else:
        os.environ["DATABASE_URL"] = secrets["PROD_DATABASE_URL"]

    return secrets
