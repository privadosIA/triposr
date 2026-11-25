# triposr-service-api Specification

## Purpose

The HTTP contract of the service (the same as the Python container's), what it says about missing weights, and the version that identifies a model.

## Requirements

### Requirement: The service answers the contract of the Python container

The service SHALL answer `GET /health`, `GET /engines`, `POST /generate`, `GET /jobs/{id}`, `GET /jobs/{id}/model` and `DELETE /jobs/{id}` with the paths, methods, status codes and JSON field names of `docker/model3d/server.py` of Veritmo: a picture (png, jpg or webp, up to 20 MB) as the body of `POST /generate?engine=triposr&detail=low|normal|high` is queued and answered `202 {id}`; a job goes `queued`, `running`, then `done` or `failed` (with `error` a sentence); `GET /jobs/{id}/model` is the `.glb` as `model/gltf-binary` or `409` until the job is done; a bearer token, when one is set, is needed by every call except `/docs` and `/openapi.yaml`. One model SHALL be made at a time and the others wait in arrival order; `/health` SHALL answer while one is made (`busy`, `queued`).

#### Scenario: A picture becomes a job
- **WHEN** a png is posted to `/generate?detail=low` and the engine is ready
- **THEN** it answers `202` with an id, and the job ends `done` with the triangles, the engine version and the detail it was made at

#### Scenario: Requests that cannot be made
- **WHEN** the engine is another one, the detail is none of the three, or the body is not a picture
- **THEN** it answers `400` with a sentence, and a body over 20 MB is `413`

#### Scenario: The description matches the service
- **WHEN** `/openapi.yaml` is read
- **THEN** it lists every call the service answers, and a test fails if they differ

### Requirement: The service says what is missing

`GET /health` and `GET /engines` SHALL report `ready: false` and name in `missing` every file of the weights folder that is not there (`EngineInfo.RequiredFiles`) and a device that does not work; they SHALL never load the model.

#### Scenario: A weights folder that is incomplete
- **WHEN** a file of the weights is missing
- **THEN** `/health` says `ready: false` and names it, and `/engines` says `ready: false` for the engine

### Requirement: What a model says made it

A job and `GET /engines` SHALL report `engine_version` as `triposr-net@<EngineInfo.PortVersion>`, and the port version SHALL change whenever a model would.

#### Scenario: The version is known before a model is made
- **WHEN** `GET /engines` is asked
- **THEN** the version is there, equal to the one a finished job reports
