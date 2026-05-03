import boto3
import os
import json
from pipeline_tracker import increment_retry, update_stage, get_document

SQS_PROCESSING_URL = os.getenv("SQS_PROCESSING_URL")
SQS_DLQ_URL        = os.getenv("SQS_DLQ_URL")
REGION             = os.getenv("AWS_REGION", "us-east-1")

sqs = boto3.client("sqs", region_name=REGION)

def send_to_processing_queue(document_id: str, filename: str, s3_key: str):
    message = {
        "document_id": document_id,
        "filename":    filename,
        "s3_key":      s3_key
    }
    sqs.send_message(
        QueueUrl=SQS_PROCESSING_URL,
        MessageBody=json.dumps(message)
    )
    print(f"[SQS] Sent to processing queue: {document_id}")


def handle_failure(document_id: str, stage: str, error: str):
    count = increment_retry(document_id)

    if count >= 3:
        update_stage(document_id, "failed",
                     status="failed",
                     message=f"Moved to DLQ after 3 retries. Last error: {error}")
        doc = get_document(document_id)
        sqs.send_message(
            QueueUrl=SQS_DLQ_URL,
            MessageBody=json.dumps({
                "document_id":  document_id,
                "filename":     doc.get("filename", "unknown"),
                "failed_stage": stage,
                "error":        error,
                "retry_count":  count
            })
        )
        print(f"[DLQ] Document {document_id} sent to dead-letter queue")
    else:
        update_stage(document_id, stage,
                     status="retrying",
                     message=f"Retry {count}/3 - {error}")
        doc = get_document(document_id)
        sqs.send_message(
            QueueUrl=SQS_PROCESSING_URL,
            MessageBody=json.dumps({
                "document_id": document_id,
                "filename":    doc.get("filename", "unknown"),
                "s3_key":      doc.get("s3_key", ""),
                "retry":       count
            })
        )
        print(f"[RETRY] Document {document_id} retry {count}/3")