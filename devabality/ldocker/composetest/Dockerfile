FROM python:3.10-alpine
WORKDIR /code
ENV FLASK_APP app.py
ENV FLASK_RUN_HOST 0.0.0.0
# proxy
# # 设置代理环境变量
# ENV HTTP_PROXY http://localhost:7890
# ENV HTTPS_PROXY http://localhost:7890
RUN apk add --no-cache gcc musl-dev linux-headers
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD ["flask", "run"]