ROOT_PASSWD="${MYSQL_ROOT_PASSWD:?Set MYSQL_ROOT_PASSWD in your environment}"
DOCKER_NAME="application-db"
REPO="${MYSQL_IMAGE:-mysql:8}"
TAG="1.0"

# Run mysql Docker container without persistance, specify root passwd
docker run -dit --rm --name ${DOCKER_NAME} \
	-v "$(pwd)/mysql_scripts":/mysql_scripts \
	--network host -e MYSQL_ROOT_PASSWORD=${ROOT_PASSWD} \
	mysql:8 --default-authentication-plugin=mysql_native_password
# Wait until docker is up
until [[ "$(docker ps -a | grep ${DOCKER_NAME})" == *"Up"* ]]; do
    echo "Waiting one second for container ${DOCKER_NAME} to start..."
    sleep 1;
done;
# Create DB and table
#docker exec -it ${DOCKER_NAME} /bin/bash -c "mysql --user=root --password=${MYSQL_ROOT_PASSWORD} < mysql_scripts/createDB.sql"
#docker exec -it ${DOCKER_NAME} /bin/bash -c "mysql --user=root --password=${MYSQL_ROOT_PASSWORD} < mysql_scripts/createUser.sql"

# Commit
# docker commit ${DOCKER_NAME} ${REPO}:${TAG}
