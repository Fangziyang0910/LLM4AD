# BehaveSim archive-wide population validation

The sample is drawn directly from every valid V10.6 archive node by evaluation
order. It does not require prior E1 profile coverage and does not stratify on
fitness.

```bash
.venv/bin/python -m experiments.behavesim_archive_validation.run prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python -m experiments.behavesim_archive_validation.run profile --workers 12
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=12 .venv/bin/python -m experiments.behavesim_archive_validation.run analyze
```
