DOCKER_NAME="inference-perf-analyzer"
MODEL_NAME="model"
RESULTS_HOST="$(pwd)/results"
RESULTS_DOCKER="/results"

# Create the results directory
mkdir -p ${RESULTS_HOST}
# Start a container with the required libraries
docker run -dit --rm --name ${DOCKER_NAME} \
	--net=host \
	-v ${RESULTS_HOST}:${RESULTS_DOCKER} \
	nvcr.io/nvidia/tritonserver:22.03-py3-sdk
# Wait until the container is ready
until [[ "$(docker ps -a | grep ${DOCKER_NAME})" == *"Up"* ]]; do
    echo "Waiting one second for container ${DOCKER_NAME} to start..."
    sleep 1;
done;
# Run the example
docker exec -it ${DOCKER_NAME} perf_analyzer -m ${MODEL_NAME} --concurrency-range 1:4 -f /results/perf.csv
