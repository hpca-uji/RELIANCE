import sys
import numpy as np
import tritonclient.grpc
import pathlib

LABEL_FILE=pathlib.Path(pathlib.Path(__file__).parent.resolve().as_posix(),"labels/chest/labels.txt")

def buildResponseJson(predictions, max, labels_dict):
    predsAndLabels = []
    for idx, label in labels_dict.items():
        predsAndLabels.append(f"{label}: {predictions[idx]}")
    
    return {
        "Predictions": predsAndLabels
    }

def inferInTriton(data, model_name="model", url="localhost:8001", verbose=False, label_file=LABEL_FILE):
    try:
        triton_client = tritonclient.grpc.InferenceServerClient(
            url=url, verbose=verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    with open(label_file) as f:
        labels_dict = {idx: line.strip() for idx, line in enumerate(f)}

    inputs = []
    outputs = []
    input_name = "input"
    output_name = "MultiLabelSmartResizeDenseNet"

    inputs.append(
        tritonclient.grpc.InferInput(input_name, data.shape, "FP32"))
    outputs.append(tritonclient.grpc.InferRequestedOutput(output_name))

    # Initialize the data
    inputs[0].set_data_from_numpy(data)
    # Run the inference
    results = triton_client.infer(model_name=model_name,
                                  inputs=inputs,
                                  outputs=outputs)

    output0_data = results.as_numpy(output_name)
    maxs = np.argmax(output0_data, axis=1)

    return buildResponseJson(output0_data[0], maxs[0], labels_dict)

def inferInTritonTLS(data, model_name="model", url="localhost:8001", verbose=False, label_file=LABEL_FILE):
    root_certificates=pathlib.Path(pathlib.Path(__file__).parent.resolve().as_posix(),"certs/ca/ca-cert.pem")
    private_key=pathlib.Path(pathlib.Path(__file__).parent.resolve().as_posix(),"certs/client/client-key.pem")
    certificate_chain=pathlib.Path(pathlib.Path(__file__).parent.resolve().as_posix(),"certs/client/client-cert.pem")
    try:
        triton_client = tritonclient.grpc.InferenceServerClient(
            url=url, verbose=verbose, ssl=True,
            root_certificates=root_certificates,
            private_key=private_key,
            certificate_chain=certificate_chain)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    with open(label_file) as f:
        labels_dict = {idx: line.strip() for idx, line in enumerate(f)}

    inputs = []
    outputs = []
    input_name = "input"
    output_name = "MultiLabelSmartResizeDenseNet"
    #output_name = "Identity"

    inputs.append(
        tritonclient.grpc.InferInput(input_name, data.shape, "FP32"))
    outputs.append(tritonclient.grpc.InferRequestedOutput(output_name))

    # Initialize the data
    inputs[0].set_data_from_numpy(data)
    # Run the inference
    results = triton_client.infer(model_name=model_name,
                                  inputs=inputs,
                                  outputs=outputs)

    output0_data = results.as_numpy(output_name)
    maxs = np.argmax(output0_data, axis=1)

    return buildResponseJson(output0_data[0], maxs[0], labels_dict)
