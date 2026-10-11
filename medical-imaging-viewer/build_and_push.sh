REPOSITORY="${IMAGE_REPOSITORY:?Set IMAGE_REPOSITORY to your image namespace}"
IMAGE_NAME="medical-imaging-viewer"
IMAGE_VERSION="${IMAGE_VERSION:-2.0}"
USERNAME="${DOCKER_USER:?Set DOCKER_USER to your registry username}"

docker login -u ${USERNAME}
docker build . -t ${REPOSITORY}/${IMAGE_NAME}:${IMAGE_VERSION}
docker push ${REPOSITORY}/${IMAGE_NAME}:${IMAGE_VERSION}
docker logout
