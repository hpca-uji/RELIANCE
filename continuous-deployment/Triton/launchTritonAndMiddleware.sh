#!/bin/bash
TRITON_MODEL_PATH="/models"

# Launch tritonserver
tritonserver \
        --model-repository=${TRITON_MODEL_PATH} \
        --model-control-mode=explicit --load-model=model \
        --grpc-use-ssl=true --grpc-use-ssl-mutual=true \
        --grpc-server-cert=certs/server/server-cert.pem \
        --grpc-server-key=certs/server/server-key.pem \
        --grpc-root-cert=certs/ca/ca-cert.pem \
        --allow-metrics=true --allow-gpu-metrics=false &
# Launch middleware
python3 -u PythonMiddleware/StorageSCP_Server.py 
