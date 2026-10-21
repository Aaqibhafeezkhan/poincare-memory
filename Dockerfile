# Minimal image to run the demo and CLI without installing anything locally.
#   docker build -t poincare-memory .
#   docker run --rm -v "$PWD/out:/out" poincare-memory demo -o /out/demo
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE CITATION.cff ./
COPY src ./src
RUN pip install --no-cache-dir . && pip install --no-cache-dir matplotlib pandas
ENTRYPOINT ["poincare-memory"]
CMD ["demo", "-o", "/tmp/demo"]
