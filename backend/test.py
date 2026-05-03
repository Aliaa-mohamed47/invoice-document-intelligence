from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification
import os

def load_model(model_path):
    print(f"Loading model from: {model_path}...")
    try:
        # استخدام TokenizerFast عشان يقرأ ملف tokenizer.json اللي في صورتك
        tokenizer = LayoutLMTokenizerFast.from_pretrained(model_path)
        
        # تحميل الموديل مع التأكد من استخدام safetensors
        model = LayoutLMForTokenClassification.from_pretrained(
            model_path, 
            use_safetensors=True
        )
        
        print("✅ Success! Model loaded correctly.")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Still an error: {e}")
        return None, None

# التأكد من المسار
MODEL_PATH = "./ai/model/saved_model"
model, tokenizer = load_model(MODEL_PATH)