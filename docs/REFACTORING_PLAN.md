# Repository refactoring plan

The repository is being migrated incrementally rather than flattened or deleted.

## Target structure

    apps/                 Deployable interactive applications
      titanic_survival_game/
    research/             Research-grade projects such as virtual_sensor
    src/                  Shared reusable Python modules
    data/                 Source datasets and generated analysis outputs
    notebooks/            Course and exploratory notebooks
    docs/                 Project documentation and provenance
    tests/                Regression and data-quality checks
    legacy/               Historical scripts preserved during migration

## Migration rules

1. Do not delete original notebooks, reports, datasets, or result images.
2. New applications must use relative paths, explicit dependencies, and a README.
3. Training code, inference code, visualizations, and UI code stay separate.
4. Every external dataset receives a provenance entry and a caveat.
5. Every model-facing feature must be available before the historical outcome.
