# Python 3.12のベースイメージを使用
FROM python:3.12-slim

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    git \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# 依存関係ファイルをコピー
COPY requirements.txt .

# Pythonパッケージをインストール
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのソースコードをコピー
COPY . .

# Streamlitのポートを公開
EXPOSE 8501

# 環境変数を設定
ENV PYTHONUNBUFFERED=1

# アプリケーションを実行
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"] 