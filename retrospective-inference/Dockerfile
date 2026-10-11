FROM python:3.12.2-slim

WORKDIR /app

RUN mkdir -p logs
RUN mkdir -p dicoms

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

# Logging
COPY logging_configs/config.json logging_configs/config.json
COPY app_logger.py app_logger.py

# Models
#COPY models-repository/*.pt models-repository/
#COPY models-repository/*.txt models-repository/

# App
COPY swipe_tags_move.py .

ENTRYPOINT [ "/bin/bash" ]
# CMD ["python3 -u swipe_tags_move.py"]
