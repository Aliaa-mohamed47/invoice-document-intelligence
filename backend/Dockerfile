FROM python:3.11-slim

WORKDIR /var/task

# تثبيت المكتبات الضرورية
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ الكود
COPY . .

# Lambda بتحتاج Handler محدد
CMD ["main.handler"]
