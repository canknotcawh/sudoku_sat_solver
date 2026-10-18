## Installation

Python 3.12 or 3.13 is recommended.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # core: SAT solvers, CP-SAT, tests, plots
pip install -r requirements-extra.txt    # optional: Gurobi, CPLEX (MIP and CP), CBC baselines
```

The packages in `requirements-extra.txt` are optional. The benchmark and the tests skip any
baseline whose library is missing. The pip editions of `gurobipy` and `cplex`/`docplex` are free
but **limited in model size** (CPLEX Community Edition, Gurobi restricted license), so the larger
grids exceed what they accept. Those runs are recorded with status `license_limit` rather than
as a solver failure. A full academic or commercial license removes the cap.
