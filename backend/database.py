import boto3
import os
from botocore.exceptions import ClientError

# إعدادات AWS من متغيرات البيئة
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
TABLE_NAME = os.getenv("DYNAMO_TABLE_NAME", "InvoicesTable")

# الاتصال بـ DynamoDB
dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
table = dynamodb.Table(TABLE_NAME)

def save_invoice_to_db(data: dict):
    """حفظ بيانات الفاتورة في DynamoDB"""
    try:
        table.put_item(Item=data)
        return True
    except ClientError as e:
        print(f"Error saving to DynamoDB: {e.response['Error']['Message']}")
        return False

def get_all_invoices_from_db():
    """جلب كل الفواتير من الجدول"""
    try:
        response = table.scan()
        return response.get('Items', [])
    except ClientError as e:
        print(f"Error fetching from DynamoDB: {e.response['Error']['Message']}")
        return []

def get_invoice_by_id(invoice_id: str):
    """جلب فاتورة محددة"""
    try:
        response = table.get_item(Key={'id': invoice_id})
        return response.get('Item')
    except ClientError as e:
        print(f"Error fetching item: {e.response['Error']['Message']}")
        return None