#!/bin/bash
#set -o errexit # abort on nonzero exit status
#set -o nounset # abort on unbound variable
#set -o pipefail # don't hide errors within pipes

stop_triton=1
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIEWER_CERTS_DIR="${VIEWER_CERTS_DIR:?Set VIEWER_CERTS_DIR to the viewer certificate directory}"
TRITON_MODELS_DIR="${TRITON_MODELS_DIR:?Set TRITON_MODELS_DIR to the inference model repository}"
TRITON_CERTS_DIR="${TRITON_CERTS_DIR:?Set TRITON_CERTS_DIR to the inference certificate directory}"
TRITON_CLIENT_CERTS_DIR="${TRITON_CLIENT_CERTS_DIR:?Set TRITON_CLIENT_CERTS_DIR to the inference client certificate directory}"
VIEWER_IMAGE="${VIEWER_IMAGE:?Set VIEWER_IMAGE to your configured viewer image}"
TRITON_IMAGE="${TRITON_IMAGE:?Set TRITON_IMAGE to your configured Triton image}"
cd "${PROJECT_DIR}"

# Stop viewer and triton
DOCKER_VIEWER_NAME="medical-imaging-viewer"
docker stop ${DOCKER_VIEWER_NAME}
docker rm ${DOCKER_VIEWER_NAME}

if [ "${stop_triton}" -eq "1" ]; then
    DOCKER_TRITON_NAME="triton-inference-server"
    docker stop ${DOCKER_TRITON_NAME}
fi
# Launch retrospective inference
DOCKER_NAME="retrospective-inference"
IMAGE="${INFERENCE_IMAGE:-retrospective-inference}"
IMAGE_TAG="1.0"

limit=7000
offset=0
models_repository="models-repository"
num_workers=1
work_queue_max_size=80
scu_wait_time=4
scu_group_size=2
if [ "${offset}" -eq "0" ]; then
    docker run -t --name ${DOCKER_NAME} --network=host \
        --env-file "${ENV_FILE:-${PROJECT_DIR}/.env}" \
        -v "${PROJECT_DIR}/results":"/app/results" \
	-v "${PROJECT_DIR}/models-repository":"/app/models-repository" \
        ${IMAGE}:${IMAGE_TAG} -c "python3 -u swipe_tags_move.py \
            --num-workers ${num_workers} \
            --work-queue-max-size ${work_queue_max_size} \
            --study-limit ${limit} \
            --study-offset ${offset} \
            --fresh-start \
            --models-repository ${models_repository} \
            --scu-wait-time ${scu_wait_time} \
            --scu-group-size ${scu_group_size}"
else
    docker run -t --name ${DOCKER_NAME} --network=host \
        --env-file "${ENV_FILE:-${PROJECT_DIR}/.env}" \
        -v "${PROJECT_DIR}/results":"/app/results" \
	-v "${PROJECT_DIR}/models-repository":"/app/models-repository" \
        ${IMAGE}:${IMAGE_TAG} -c "python3 -u swipe_tags_move.py \
            --num-workers ${num_workers} \
            --work-queue-max-size ${work_queue_max_size} \
            --study-limit ${limit} \
            --study-offset ${offset} \
            --models-repository ${models_repository} \
            --scu-wait-time ${scu_wait_time} \
            --scu-group-size ${scu_group_size}"
fi

#docker rm ${DOCKER_NAME} 

# Reinitialize viewer and triton
docker run -d --name ${DOCKER_VIEWER_NAME} \
        -v viewer-volume:"/usr/src/app/data" \
        -v "${VIEWER_CERTS_DIR}":"/usr/src/app/src/certs" \
        --network host \
        --ulimit core=0 \
        --log-opt max-size=30m \
        --log-opt max-file=3 \
        "${VIEWER_IMAGE}" npm start

if [ "${stop_triton}" -eq "1" ]; then
    docker run -d --name ${DOCKER_TRITON_NAME} --rm \
            -v "${TRITON_MODELS_DIR}":"/models" \
            -v "${TRITON_CERTS_DIR}":"/app/certs" \
            -v "${TRITON_CLIENT_CERTS_DIR}":"/app/PythonMiddleware/certs" \
            --network host \
            "${TRITON_IMAGE}"
fi
