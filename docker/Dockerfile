FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
EXPOSE 8501
ENV LLM_BACKEND=OLLAMA
ENV OLLAMA_HOST=http://localhost:11434
ENV OLLAMA_MODEL=llama3
CMD ["streamlit", "run", "src/app_streamlit.py", "--server.address=0.0.0.0", "--server.port=8501"]
