# <REDACTED_PROJECT_NAME> DICOM viewer and gateway

This Node.js/TypeScript service provides the authenticated web entry point for the OHIF viewer and translates DICOMweb requests to DIMSE operations against configured PACS peers. It also supports radiologist labeling and image discarding through the application database.

## Code map

- `src/app.ts`: creates the Fastify HTTPS server, loads runtime settings, configures JWT/session cookies and MySQL, registers routes, and starts the DIMSE listener.
- `src/routes/`: login/contact routes and authenticated labeling/discard endpoints.
- `src/routesohif/`: authenticated DICOMweb endpoints used by the viewer.
- `src/controllers/`: login, contact, label, and discard request handlers.
- `src/dimse/`: DICOM operations. `findData.ts` handles search; `fetchMove.ts` and `fetchGet.ts` retrieve studies; `fetchMeta.ts` and `parseMeta.ts` handle metadata; `wadoRs.ts` and `wadoUri.ts` serve image content; `store.ts` receives C-STORE; `sendEcho.ts` checks peer connectivity; `compressFile.ts` manages transfer compression; `tags.ts` and `querLevel.ts` define tag/query behavior.
- `src/db/`: database repositories and application data interfaces.
- `src/utils/`: configuration loading, DICOM decoding/file handling, hashing, and object-merge helpers.
- `src/views/`: EJS login, contact, and shared page fragments.
- `public/`: built OHIF viewer assets and static resources.
- `config/default.json`: DICOM source and PACS peer settings.
- `config/default2.json`: alternate DICOM configuration profile.
- `config/config.env`: database, cookie, and JWT settings.

## Local setup

Use a supported Node.js version, then install dependencies and compile:

```sh
npm ci
npm run build
```

Modify: `config/config.env`, replace the placeholders in that file and in `config/default.json` and `config/default2.json` with deployment-specific settings. The database account must be provisioned separately. Set a unique random JWT secret and `websocketToken`; do not commit real credentials or tokens. Some native addons have been redacted, which may cause errors during execution. Native addons will be replaced upon acceptance.

The server reads HTTPS keys at `src/certs/server.key` and `src/certs/server.crt`, and session keys beneath `src/certs/sessions/`. These files are excluded from version control. Generate local-only material with `npm run https-certs` and `npm run session-certs`, or provision trusted certificates/keys through your deployment secret manager. The self-signed certificate helper is for development only.

Start the service with:

```sh
npm start
```

`npm run format` formats TypeScript sources. `build_and_push.sh` builds and pushes an image; configure `IMAGE_REPOSITORY` and `DOCKER_USER` in your shell before running it. Docker authentication remains interactive and must not be stored in source files.

## Privacy and clinical-use notice

Deployment addresses, database passwords, JWT/websocket tokens, and embedded certificate material have been replaced with placeholders or removed. Do not store patient-identifiable DICOM content, credentials, or private keys in this repository. This software is a technical viewer/workflow component and does not provide a clinical diagnosis.