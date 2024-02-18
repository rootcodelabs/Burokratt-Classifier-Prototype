FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r /app/requirements.txt
### Note requirements.txt file is not yet completed. 
EXPOSE 8000
CMD ["python", "core_api.py"]
