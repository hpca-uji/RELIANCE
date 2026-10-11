DOCKER_NAME="dicom-mock-client"
IMAGE="python-middleware"
IMAGE_TAG="1.0"
MOCK_DICOM_FILE="$(realpath "${MOCK_DICOM_FILE:?Set MOCK_DICOM_FILE to a de-identified DICOM file}")"

# Build Docker image
docker build --network=host --rm -t ${IMAGE}:${IMAGE_TAG} -f Dockerfile .

# Run Docker container and the python middleware
docker run -dit --name ${DOCKER_NAME} \
        --network host \
        --volume "${MOCK_DICOM_FILE}:/tmp/mock.dcm:ro" \
        --env MOCK_DICOM_FILE=/tmp/mock.dcm \
        --env "DICOM_DEST_HOST=${DICOM_DEST_HOST:-localhost}" \
        --env "DICOM_DEST_PORT=${DICOM_DEST_PORT:-104}" \
        ${IMAGE}:${IMAGE_TAG}
# Wait until docker is exited
until [[ "$(docker ps -a | grep ${DOCKER_NAME})" == *"Exited"* ]]; do
    echo "Waiting one second for container ${DOCKER_NAME} to finish..."
    sleep 1;
done;
# Verify Execution was Successful
docker logs ${DOCKER_NAME}
# Delete stopped container
docker rm ${DOCKER_NAME}
