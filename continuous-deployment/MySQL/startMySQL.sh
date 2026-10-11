DOCKER_NAME="application-db"
VOLUME_NAME="db-volume"
IMAGE="${MYSQL_IMAGE:-application-db:local}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${SCRIPT_DIR}"

docker build --rm -t "${IMAGE}" -f Dockerfile .

# Create a volume
docker volume create ${VOLUME_NAME}
# Run mysql Docker container with persistance, specify root passwd
docker run -dit --rm --name ${DOCKER_NAME} \
	-v ${VOLUME_NAME}:/var/lib/mysql \
	--network host \
	--env-file "${PROJECT_ROOT}/.env" \
	${IMAGE} --default-authentication-plugin=mysql_native_password
# Wait until docker is up
until [[ "$(docker ps -a | grep ${DOCKER_NAME})" == *"Up"* ]]; do
    echo "Waiting one second for container ${DOCKER_NAME} to start..."
    sleep 1;
done;
# Delete volume
# docker volume rm ${VOLUME_NAME}
