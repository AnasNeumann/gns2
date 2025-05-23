import argparse
import os

# ##################################################
# =*= CODE TO GENERATE JOBS FOR THE EXACT SOLVER =*=
# ##################################################
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"

START_IDX: int = 151
END_IDX: int = 200
SIZES: list[str] = ['s', 'm', 'l', 'xl', 'xxl', 'xxxl']
MINUTES: list[int] = [10, 10, 10, 10, 10, 10]
HOURS: list[int] = [2, 6, 9, 12, 20, 30]
MEMORY: list[int] = [30, 90, 135, 180, 300, 450]

'''
    TEST WITH
    python jobs/exact_builder.py --account=x --parent=y --mail=x@mail.com
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EPSIII job builder")
    parser.add_argument("--account", help="Compute Canada Account", required=True)
    parser.add_argument("--parent", help="Compute Canada Parent Account", required=True)
    parser.add_argument("--mail", help="Compute Canada Email Adress", required=True)
    args = parser.parse_args()
    BASIC_PATH = "/home/"+args.account+"/projects/def-"+args.parent+"/"+args.account+"/gns2/"
    for size_id, size in enumerate(SIZES):
        os.makedirs(f"./scripts/exact/{size}/", exist_ok=True) 
        memory = MEMORY[size_id] - 2
        for instance in range(START_IDX, END_IDX+1):
            f = open(f"./scripts/exact/{size}/{instance}.sh", "w+")
            f.write("#!/bin/bash\n")
            f.write("#SBATCH --nodes 1\n")
            f.write(f"#SBATCH --time={HOURS[size_id]}:{MINUTES[size_id]}:00\n")
            f.write(f"#SBATCH --mem={MEMORY[size_id]}G\n")
            f.write(f"#SBATCH --cpus-per-task=32\n")
            f.write(f"#SBATCH --account=def-{args.parent}\n")
            f.write(f"#SBATCH --mail-user={args.mail}\n")
            f.write("#SBATCH --mail-type=FAIL\n")
            f.write(f"#SBATCH --output={BASIC_PATH}data/out/exact_{size}_{instance}.out\n")  
            f.write("module load python/3.12\n")
            f.write("virtualenv --no-download $SLURM_TMPDIR/env\n")
            f.write("source $SLURM_TMPDIR/env/bin/activate\n")
            f.write("pip install --upgrade pip --no-index\n")
            f.write(f"pip install {BASIC_PATH}wheels/protobuf-5.28.3-*.whl\n")
            f.write(f"pip install {BASIC_PATH}wheels/immutabledict-4.2.0-*.whl\n")
            f.write("pip install --no-index -r "+BASIC_PATH+"requirements_or.txt\n")
            f.write(f"python {BASIC_PATH}exact_solver.py --mode=prod --size={size} --id={instance} --time={HOURS[size_id]} --memory={memory} --path="+BASIC_PATH+" \n")
            f.write("deactivate\n")
            f.close()
