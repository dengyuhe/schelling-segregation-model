# Original code

The code as originally written for the project, kept for provenance:

- `segregationModel.py`: the original model
- `dataGathering.py`: the original sweep of equal minimum tolerances (0 to 0.5)

The restructured package in `schelling/` replaces it. Running `Model.run(..., legacy_cache=True)`
reproduces this code's output exactly, run for run (checked in `tests/test_model.py`). The default
mode fixes a bug in how the original tracked unsatisfied agents (see the main README).
