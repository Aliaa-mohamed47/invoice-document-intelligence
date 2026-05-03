import boto3
import uuid
from datetime import datetime, timezone
from botocore.exceptions import ClientError
import os

# ── Config ──────────────────────────────────────────────
TABLE_NAME = os.getenv("DYNAMODB_TABLE", "invoice_pipeline")
REGION     = os.getenv("AWS_REGION", "us-east-1")

dynamodb = boto3.resource("dynamodb", region_name=REGION)
table    = dynamodb.Table(TABLE_NAME)

# الـ stages الممكنة
STAGES = ["uploaded", "extracting", "validating", "storing", "completed", "failed"]

# ── Create ───────────────────────────────────────────────
def create_document_record(filename: str, s3_key: str) -> str:
    """
    بتسجل invoice جديدة في DynamoDB.
    بترجع document_id اللي هتستخدمه في كل الـ calls التانية.
    """
    document_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    table.put_item(Item={
        "document_id":   document_id,
        "filename":      filename,
        "s3_key":        s3_key,
        "current_stage": "uploaded",
        "version":       1,
        "retry_count":   0,
        "created_at":    now,
        "updated_at":    now,
        "history": [
            {
                "stage":     "uploaded",
                "status":    "success",
                "timestamp": now,
                "message":   "Document received and stored in S3"
            }
        ]
    })

    print(f"[TRACKER] Created record for {filename} → ID: {document_id}")
    return document_id


# ── Update Stage ─────────────────────────────────────────
def update_stage(document_id: str, new_stage: str,
                 status: str = "success", message: str = "") -> bool:
    """
    بتحدث الـ stage الحالية وبتضيف entry في الـ history.
    status: "success" | "failed" | "retrying"
    """
    if new_stage not in STAGES:
        raise ValueError(f"Invalid stage: {new_stage}. Must be one of {STAGES}")

    now = datetime.now(timezone.utc).isoformat()

    history_entry = {
        "stage":     new_stage,
        "status":    status,
        "timestamp": now,
        "message":   message or f"Stage {new_stage} - {status}"
    }

    try:
        table.update_item(
            Key={"document_id": document_id},
            UpdateExpression="""
                SET current_stage = :stage,
                    updated_at    = :now,
                    #hist          = list_append(#hist, :entry)
            """,
            ExpressionAttributeNames={"#hist": "history"},
            ExpressionAttributeValues={
                ":stage": new_stage,
                ":now":   now,
                ":entry": [history_entry]
            }
        )
        print(f"[TRACKER] {document_id} → {new_stage} ({status})")
        return True

    except ClientError as e:
        print(f"[TRACKER ERROR] {e.response['Error']['Message']}")
        return False


# ── Retry Logic ──────────────────────────────────────────
def increment_retry(document_id: str) -> int:
    """
    بتزود الـ retry_count بـ 1.
    لو وصل 3 → بترجع -1 كإشارة إن الـ document هتروح DLQ.
    """
    try:
        response = table.update_item(
            Key={"document_id": document_id},
            UpdateExpression="SET retry_count = retry_count + :inc",
            ExpressionAttributeValues={":inc": 1},
            ReturnValues="UPDATED_NEW"
        )
        count = int(response["Attributes"]["retry_count"])
        print(f"[TRACKER] Retry count for {document_id}: {count}")
        return count

    except ClientError as e:
        print(f"[TRACKER ERROR] {e.response['Error']['Message']}")
        return -1


# ── Get Record ───────────────────────────────────────────
def get_document(document_id: str) -> dict | None:
    """
    بتجيب الـ record الكاملة من DynamoDB.
    """
    try:
        response = table.get_item(Key={"document_id": document_id})
        return response.get("Item")
    except ClientError as e:
        print(f"[TRACKER ERROR] {e.response['Error']['Message']}")
        return None


# ── Get History ──────────────────────────────────────────
def get_history(document_id: str) -> list:
    """
    بترجع الـ processing history كاملة لـ document معين.
    """
    doc = get_document(document_id)
    if doc:
        return doc.get("history", [])
    return []