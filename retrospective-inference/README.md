# <REDACTED_PROJECT_NAME> retrospective inference

This project retrieves previously registered DICOM instances from a PACS, applies TorchScript image-classification models, and writes per-model tab-separated result files. It also includes a utility for formatting an older MySQL prediction export.

## Processing flow and code map

- `swipe_tags_move.py` contains the pipeline:
  - `DBPool` connects to MySQL using environment variables.
  - `DBImplementation.read_studies()` selects records linked to predictions for the configured reviewer account, excludes discarded DICOMs, orders by `DicomID`, and applies a caller-supplied limit and offset.
  - `SCUClient` requests each selected instance from the configured PACS with DICOM C-MOVE.
  - `SCPServer` receives the C-STORE objects, runs each `.pt` TorchScript model, removes its temporary DICOM file, and appends predictions to a tab-separated file in `results/`.
- `models-repository/` is meant to hold TorchScript models and the model-label file. The used TorchScript models will be made available upon acceptance.
- `launchRetrospectiveInference.sh` builds and starts the Docker image.
- `cron_retrospective_inference.sh` is a deployment orchestration script; it also stops/restarts configured viewer and Triton containers.
- `extract_predictions.sql` queries labels/results from MySQL.
- `format_predictions_to_csv.py` converts the legacy SQL text export into a disease-column table.
- `app_logger.py` and `logging_configs/config.json` configure queued structured logging.

## Configuration and run

Modify `.env`:  and replace every placeholder with values approved for your environment. Set `REVIEWER_USER_ID` to the authorized reviewer account in the target database. The database account needs read access to the study, DICOM, and radiologist-prediction tables. Configure `DICOM_PEER_HOST`, `DICOM_PEER_AE`, and `DICOM_PEER_PORT` for the PACS, and ensure the PACS can send C-STORE responses back to the inference host on port `11113` using the configured receiver AE title `INFERENCE_SCP`.

Build and launch from this directory:

```sh
# Edit .env with local database and PACS settings before use.
docker build --rm -t retrospective-inference:1.0 -f Dockerfile .
./launchRetrospectiveInference.sh
```

`launchRetrospectiveInference.sh` defaults to a 13-study batch; the Python entry point defaults to 12. Use its `--study-limit`, `--study-offset`, `--num-workers`, `--work-queue-max-size`, `--scu-wait-time`, and `--scu-group-size` options to control a run. `--fresh-start` overwrites prior output, so back up any results you need before using it. Generated CSV/TSV files, logs, temporary DICOMs, and `.env` are intentionally excluded from version control.

The optional `cron_retrospective_inference.sh` also stops the running viewer and Triton containers before inference and restarts them afterward. It is deployment-specific; set `VIEWER_IMAGE`, `TRITON_IMAGE`, `VIEWER_CERTS_DIR`, `TRITON_MODELS_DIR`, `TRITON_CERTS_DIR`, and `TRITON_CLIENT_CERTS_DIR` before using it.

## Privacy and clinical-use notice

Patient-linked prediction exports, sample DICOMs, private credentials, deployment IP addresses, and personal filesystem paths have been removed or replaced with configuration placeholders. Do not add identifiable DICOM data, keys, or live credentials to this repository. Model outputs are experimental and are not a substitute for clinical interpretation.