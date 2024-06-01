# 베이스 이미지 선택 (Python 3.9)
FROM python:3.9

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 파일들 복사
COPY . /app

# 필요한 패키지 설치
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Flask 앱 실행
CMD ["gunicorn", "-b", "0.0.0.0:5000", "model1:app"]
