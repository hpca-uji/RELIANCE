FROM python:3

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt
COPY src/ .

ENTRYPOINT [ "python3" ]
CMD ["-u", "StorageSCU_Client.py"]
