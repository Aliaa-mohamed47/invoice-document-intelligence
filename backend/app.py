import json
import base64

from lark import logger
import boto3
import os
import torch # تأكد من استيراده للتحقق من الـ CPU/GPU

# استيراد الوظائف الخاصة بك
try:
    from ai.model.pipeline import load_model, extract_entities
except ImportError:
    # هذا يساعدك لو كانت المسارات مختلفة داخل الحاوية
    from pipeline import load_model, extract_entities

MODEL_PATH = os.environ.get("MODEL_PATH", "/var/task/ai/model/saved_model")

# تحميل الموديل مرة واحدة فقط عند تشغيل الحاوية
model, tokenizer = None, None
try:
    if os.path.exists(MODEL_PATH):
        model, tokenizer = load_model(MODEL_PATH)
        # توفير الذاكرة: نستخدم المعالج العادي في Lambda
        if model:
            model.to("cpu")
    else:
        print(f"Model path not found: {MODEL_PATH}")
except Exception as e:
    print(f"Initialization error: {e}")

def lambda_handler(event, context):
    if not model or not tokenizer:
        return {"statusCode": 500, "body": json.dumps({"error": "Model not loaded on cold start"})}

    try:
        # التعامل مع الاحتمالات المختلفة لمدخلات Lambda (API Gateway vs Direct)
        body = event.get('body', event)
        if isinstance(body, str):
            body = json.loads(body)

        if 'image' not in body:
            return {"statusCode": 400, "body": "Missing 'image' base64 payload."}

        image_bytes = base64.b64decode(body['image'])

        # استدعاء AWS Textract (تأكد من تفعيل صلاحية Textract في IAM Role)
        textract = boto3.client('textract')
        response = textract.detect_document_text(Document={'Bytes': image_bytes})

        tokens = []
        bboxes = []
        img_width, img_height = 1000, 1000

        for item in response.get('Blocks', []):
            if item['BlockType'] == 'WORD':
                tokens.append(item['Text'])
                b = item['Geometry']['BoundingBox']
                # تحويل الإحداثيات لـ Integers بين 0-1000 لضمان توافق LayoutLM
                bboxes.append([
                    int(b['Left'] * img_width),
                    int(b['Top'] * img_height),
                    int((b['Left'] + b['Width']) * img_width),
                    int((b['Top'] + b['Height']) * img_height)
                ])

        # التنفيذ (Inference)
        results = extract_entities(tokens, bboxes, model, tokenizer, img_width, img_height)

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*' # ضروري للـ Frontend
            },
            'body': json.dumps(results)
        }

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)})
        }