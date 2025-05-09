# Engineer-To-Order (ETO) Graph Neural Scheduling (GNS) Project [version 2]
A Graph Attention Network (GAT) to schedule jobs in an ETO manufacturing environment, trained with a Multi-Agent version of the Deep Q-Learning algorithm.

## Refer to this repository in scientific documents
`Neumann, Anas (2025). A hyper-graph neural network trained with multi-agents deep Q-learning to schedule engineer-to-order projects *GitHub repository: https://github.com/AnasNeumann/gns2*.`

```bibtex
    @misc{HGNS,
    author = {Anas Neumann},
    title = {A hyper-graph neural network trained with multi-agents deep Q-learning to schedule engineer-to-order projects},
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

## Generate a dataset of training and test instances
```bash
python generators/instance_generator.py --debug=false --train=150 --test=50
```

## Use CP solver to get optimal values
```bash
python solvers/exact_solver.py --size=s --id=151 --mode=test --path=./ --time=1 --memory=8
```

## Build and run SLURM jobs to execute the exact solver on DRAC super-computers
```bash
python jobs/exact_builder.py --account=x --parent=y --mail=x@mail.com
bash jobs/scripts/0_run_purge.sh
bash jobs/scripts/1_run_all.sh exact_s
```

## Use the GNS2 solver (inference mode) to get good heuristic values
```bash
python main.py --train=false --target=true --path=./ --mode=test --version=1 --itrs=0 --size=s --id=151 # one instance only
python main.py --train=false --target=false --path=./ --mode=prod --version=1 --itrs=0 # all test instances
```

## Train the GNS2 solver using e-greedy DQN
```bash
python main.py --train=true --mode=prod --version=1 --interactive=true --path=./
```

## Build and run SLURM jobs to train the GNS2 solver on DRAC super-computers
```bash
jobs/gns_builder.py --account=x --parent=y --mail=x@mail.com --time=20 --memory=187 --cpu=16 --version=1 --itrs=0
sbatch jobs/scripts/train_gns.sh
```

## Analyze the final results
```bash
python analysis/results_analysis.py --path=./ --last=9
```