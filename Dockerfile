FROM python:3.12-slim

# Ставим системные библиотеки: gcc и libpq-dev нужны для psycopg2, 
# libgomp1 нужен для scikit-learn и onnxruntime
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# ВАЖНО: ставим CPU-версию PyTorch (чтобы образ не весил 10 ГБ)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Ставим остальные библиотеки
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код
COPY . .

# Запускаем главный файл
CMD ["sh", "-c", "python ingest.py && python main.py"]