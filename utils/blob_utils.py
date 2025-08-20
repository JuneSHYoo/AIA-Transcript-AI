import datetime
from pathlib import Path
import urllib.parse
from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas

##################################
## azure 스토리지 업로드 + SAS 생성  ##
##################################
def upload_and_get_sas(local_path: str, account: str, key: str, container: str) -> str:
    p = Path(local_path)
    if not p.exists():
        raise FileNotFoundError(p)

    conn = f"DefaultEndpointsProtocol=https;AccountName={account};AccountKey={key};EndpointSuffix=core.windows.net"
    bsc = BlobServiceClient.from_connection_string(conn)
    container_client = bsc.get_container_client(container)
    try:
        container_client.create_container()
    except Exception:
        pass

    blob = container_client.get_blob_client(p.name)
    with open(p, "rb") as f:
        blob.upload_blob(f, overwrite=True)

    # timezone-aware UTC
    expiry = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=24)
    sas = generate_blob_sas(
        account_name=account,
        container_name=container,
        blob_name=p.name,
        account_key=key,
        permission=BlobSasPermissions(read=True),
        expiry=expiry,
    )
    return f"https://{account}.blob.core.windows.net/{container}/{urllib.parse.quote(p.name)}?{sas}"
