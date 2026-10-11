DOCKER_NAME="triton-inference-server"
TRITON_MODEL_PATH="/models"
HTTP_PORT=8000
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"

# Build Docker image
docker build --network=host --rm -t triton-inference:1.0 -f Dockerfile-triton .

# Run Docker container and the python middleware
docker run -dit --name ${DOCKER_NAME} --rm \
        --env-file "${ENV_FILE}" \
        -v "$(pwd)/models-repository":${TRITON_MODEL_PATH} \
        --network host \
        triton-inference:1.0
docker run -dit --name ${DOCKER_NAME} --rm \
	-v "$(pwd)/models-repository":${TRITON_MODEL_PATH} \
	-p ${HTTP_PORT}:${HTTP_PORT} -p 8001:8001 -p 8002:8002 -p 11112:11112 -p 3306:3306 \
	triton-inference:1.0
# Wait until docker is up
until [[ "$(docker ps -a | grep ${DOCKER_NAME})" == *"Up"* ]]; do
    echo "Waiting one second for container ${DOCKER_NAME} to start..."
    sleep 1;
done;
# Run the triton server inside the container
docker exec -dit ${DOCKER_NAME} tritonserver \
       --model-repository=${TRITON_MODEL_PATH} \
       --model-control-mode=explicit --load-model=model \
       --grpc-use-ssl=true --grpc-use-ssl-mutual=true \
       --grpc-server-cert=certs/server/server-cert.pem \
       --grpc-server-key=certs/server/server-key.pem \
       --grpc-root-cert=certs/ca/ca-cert.pem \
	--allow-metrics=true --allow-gpu-metrics=false
# Wait until tritonserver service is up
while [[ "$(curl -s -o /dev/null -w ''%{http_code}'' localhost:${HTTP_PORT}/v2/health/ready)" != "200" ]]; do 
    echo "Waiting five seconds for the server to become ready for inference..."
    sleep 5;
done
# Verify Triton Is Running Correctly
curl -v -H "Connection: close" localhost:${HTTP_PORT}/v2/health/ready