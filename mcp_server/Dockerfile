FROM python:3.11-slim

WORKDIR /app

COPY mcp_server/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY mcp_server/server.py mcp_server/server.py

EXPOSE 8001

CMD ["python", "mcp_server/server.py"]
