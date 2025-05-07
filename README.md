# Engineer-To-Order (ETO) Graph Neural Scheduling (GNS) Project [version 2]
A Graph Attention Network (GAT) to schedule jobs in an ETO manufacturing environment, trained with a Multi-Agent version of the Deep Q-Learning algorithm.

## Refer to this repository in scientific documents
`Neumann, Anas (2025). A hyper-graph neural network trained with multi-Agent deep Q-learning to schedule engineer-to-order projects *GitHub repository: https://github.com/AnasNeumann/gns2*.`

```bibtex
    @misc{HGNS,
    author = {Anas Neumann},
    title = {A hyper-graph neural network trained with multi-Agent deep Q-learning to schedule engineer-to-order projects},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/AnasNeumann/gns2}},
    commit = {main}
    }
```

## Locally try the project
1. `python -m venv ./gns2_env`
2. `source ./gns2_env/bin/activate`
3. `pip install --upgrade -r requirements/gns_wheels.txt`
4. CHOOSE EITHER GNS_SOLVER, EXACT_SOLVER, INSTANCE_GENERATOR, or RESULTS_ANALYS (_see bellow for the rest_)
5. `desactivate`