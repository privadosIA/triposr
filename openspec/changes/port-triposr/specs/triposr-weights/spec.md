## ADDED Requirements

### Requirement: The weights are read as the container has them

The library SHALL read the weights from the folder layout of the container's `/ckpt` (`triposr/config.yaml, triposr/model.ckpt, triposr/dino-vitb16-config.json, rembg/u2net.onnx`) with no Python and no download, and SHALL refuse a file that is not the checkpoint it expects naming the key that is missing or left over. The C# modules SHALL carry the names of the checkpoint's keys so that loading needs no mapping.

#### Scenario: The strict load
- **WHEN** the checkpoint is loaded into the modules
- **THEN** every key is used once and none is left over; a key that does not fit is named

#### Scenario: Nothing is downloaded
- **WHEN** the service starts with the weights folder and no network
- **THEN** it loads the model and makes a model

### Requirement: Checkpoints are never in the repository

The repository SHALL hold no weights; tests that need them SHALL be skipped, saying why, unless `TRIPOSR_CKPT` points to them, and fixtures SHALL be checked against a sha256 table before use.

#### Scenario: A clone with no weights
- **WHEN** the tests run on a clone with no weights
- **THEN** they pass, and the ones that need the weights say they were skipped and why
