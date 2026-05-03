FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.11

WORKDIR ${LAMBDA_TASK_ROOT}

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./

ENV PYTHONPATH=${LAMBDA_TASK_ROOT}
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["main.handler"]