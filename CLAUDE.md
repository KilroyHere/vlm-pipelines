# Working in this repo

- Read `docs/contracts.md` before touching geometry. Frames: meters, Z-up, yaw CCW,
  asset front = local +Y (ADR-0001).
- `src/roomlayout/schema/` is the contract. Don't change it without bumping
  `SCHEMA_VERSION` and re-running `python -m roomlayout.schema.export`.
- Stage packages (`solver`, `assets`, ...) import `roomlayout.schema` only, never each other.
- Never import research code (SAM 3D, GOR-IS); call it out-of-process (ADR-0004).
- Before committing: `.venv/bin/pytest`, `.venv/bin/ruff check .`, `.venv/bin/ruff format --check .`
- Record significant decisions as a new ADR in `docs/adr/`, and update `docs/PROJECT_PLAN.md` status.
- Don't commit large binaries (meshes, datasets); see `.gitignore`.
