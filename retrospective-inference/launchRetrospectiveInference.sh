DOCKER_NAME="retrospective-inference"
IMAGE="${INFERENCE_IMAGE:-retrospective-inference}"
IMAGE_TAG="1.0"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${PROJECT_DIR}/.env}"
cd "${PROJECT_DIR}"

# Build Docker image
docker build --network=host --rm -t ${IMAGE}:${IMAGE_TAG} -f Dockerfile .

# Run Docker container and the python middleware
limit=13
offset=0
models_repository="models-repository"
num_workers=1
work_queue_max_size=80
scu_wait_time=4
scu_group_size=2
docker run -d --name "${DOCKER_NAME}" \
	--env-file "${ENV_FILE}" \
	--net=host \
	-v "${PROJECT_DIR}/results":"/app/results" \
	-v "${PROJECT_DIR}/models-repository":"/app/models-repository" \
	"${IMAGE}:${IMAGE_TAG}" -c "python3 -u swipe_tags_move.py \
		--num-workers ${num_workers} \
		--work-queue-max-size ${work_queue_max_size} \
		--study-limit ${limit} \
		--study-offset ${offset} \
		--models-repository ${models_repository} \
		--scu-wait-time ${scu_wait_time} \
		--scu-group-size ${scu_group_size}"
