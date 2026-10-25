FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY vectorprep ./vectorprep
RUN pip install --no-cache-dir ".[plot]" && vectorprep build-native
ENTRYPOINT ["vectorprep"]
CMD ["bench", "--quick", "--out", "/tmp/bench"]
