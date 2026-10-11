# <REDACTED_PROJECT_NAME> inference service

This project contains the DICOM receiver, image preprocessing, Triton inference service, MySQL persistence, model-conversion utilities, and a small DICOM sender for integration checks.

## Processing flow

1. `Triton/PythonMiddleware/StorageSCP_Server.py` accepts DICOM C-STORE associations on port `104`.
2. `DicomPreprocessor.py` converts pixel data to a normalized 1024 x 1024 tensor using `means/dataset_mean.txt`.
3. `grpcTritonClient.py` sends the tensor to the Triton model over gRPC with mutual TLS. `httpTritonClient.py` contains the corresponding HTTP client helper.
4. `DB/dbImplementation.py` stores model output and acquisition/processing timestamps in MySQL.

## Repository map

- `docker-compose.yaml`, `startServices.sh`: start the MySQL and Triton services.
- `MySQL/mysql_scripts/`: schema and example database-user setup.
- `Triton/models-repository/`: Store models to test here.
- `Triton/PythonMiddleware/`: DICOM C-STORE receiver, preprocessing, inference clients, and database integration.
- `Triton/TLS/`, `certificates/`: locations and helper script for locally generated TLS material. Private keys and generated certificates are intentionally not included; generate and provision your own before use.
- `MockClient/`: DICOM C-STORE sender and its Docker setup. Supply a de-identified DICOM file using `MOCK_DICOM_FILE`; clinical sample files are not distributed in this repository.
- `PerfAnalyzer/`: Triton performance-analyzer launcher and output directory.

## Configuration and startup

Modify `.env`: Set unique values for `MYSQL_ROOT_PASSWORD`, `DB_USER`, `DB_PWD`, and the database connection variables in a local, unshared environment before starting services. Configure MySQL with the same application-user credentials. Set `<DATABASE_USER>` and replace the password placeholder in `MySQL/mysql_scripts/createUser.sql` with the same `DB_PWD`. Set `DB_NAME` to the database used by your deployment. Do not commit real credentials.

Generate TLS certificates and keys for your deployment and place them at the paths expected under `Triton/TLS/` and `Triton/PythonMiddleware/certs/`. Do not reuse credentials from another installation.

The service uses host networking and the DICOM receiver listens on privileged port 104 by default with the generic called AE title `INFERENCE_SCP`. Configure the PACS return destination to match. Adjust deployment privileges, ports, and firewall rules for your environment. Model files are required in the Triton model repository. Before executing any `.sh` files, revise and fill them carefully.

## Data and clinical-use notice

The former mock DICOM images, radiology reports, generated cohort predictions, and TLS material have been removed because they contained identifying or deployment-specific data. Use only appropriately de-identified test data. Inference output is not a diagnosis and must not be used as a substitute for clinical judgment.