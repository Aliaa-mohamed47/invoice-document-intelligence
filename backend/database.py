import boto3
import os
from botocore.exceptions import ClientError

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
TABLE_NAME = os.getenv("DYNAMO_TABLE_NAME", "InvoicesTable")

dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
table = dynamodb.Table(TABLE_NAME)

def save_invoice_to_db(data: dict):
    try:
        table.put_item(Item=data)
        return True
    except ClientError as e:
        print(f"Error saving to DynamoDB: {e.response['Error']['Message']}")
        return False

def get_all_invoices_from_db():
    try:
        response = table.scan()
        return response.get('Items', [])
    except ClientError as e:
        print(f"Error fetching from DynamoDB: {e.response['Error']['Message']}")
        return []

def get_invoice_by_id(invoice_id: str):
    try:
        response = table.get_item(Key={'id': invoice_id})
        return response.get('Item')
    except ClientError as e:
        print(f"Error fetching item: {e.response['Error']['Message']}")
        return None

def delete_invoice_from_db(invoice_id: str):
    try:
        table.delete_item(Key={'id': invoice_id})
        return True
    except ClientError as e:
        print(f"Error deleting: {e}")
        return False    