## ADDED Requirements

### Requirement: Every stage matches the reference

Each stage of the model (see `docs/porting-plan.md`) SHALL be tested against fixtures of the golden run, in fp32, with `max_abs <= 1e-4` and `cos_sim >= 0.9999`, so that a difference is found at the stage where it starts; a change that touches the model SHALL NOT be archived without its parity test.

#### Scenario: A stage
- **WHEN** a stage runs on the fixtures' input
- **THEN** its output is within the thresholds of the fixture of the golden run

### Requirement: The model is the one the container makes

Over the five reference pictures, the model SHALL match the container's: the triangles count within 1% at each detail, the same bounding box within 1% of its size and a Chamfer distance below 1e-3 of it, upright (Y up) with its origin at the middle of its base, its colours kept, no more than 500,000 triangles (made again at the next lower detail when it passes them, saying so in `detail_used`), and a `.glb` that the orchestrator reads (`blender.ParseModel`).

#### Scenario: The five pictures
- **WHEN** the five pictures go through the service at `normal` detail
- **THEN** each model matches the container's within those limits and stands on the floor of a scene

#### Scenario: Too many triangles
- **WHEN** a model passes 500,000 triangles at the detail asked
- **THEN** it is made again at the next lower detail and `detail_used` says which
