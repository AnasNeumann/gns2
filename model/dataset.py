import os
import pickle
import random

from model.instance import Instance
from conf import *
from tools.common import directory

# #######################################
# =*= DATASET MANAGEMENT DATA-CLASSES =*=
# #######################################
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"

class Dataset:
    def __init__(self, base_path: str):
        self.train_instances: list[Instance] = []
        self.test_instances: list[Instance] = []
        self.base_path: str = base_path+directory.instances

    def random_one(self) -> Instance:
        idx: int = random.randint(0, len(self.train_instances)-1)
        return self.train_instances[idx]

    def load_one(self, size: str, id: str) -> Instance:
        print(f"Loading instance {id} (of size={size})...")
        with open(self.base_path+'/test/'+size+'/instance_'+id+'.pkl', 'rb') as file:
            instance: Instance = pickle.load(file)
            return instance

    def load_training_instances(self, version: int):
        for size in TRAINING_SIZES[version-1]:
            complete_path = self.base_path+'/train/'+size+'/'
            for i in os.listdir(complete_path):
                if i.endswith('.pkl'):
                    file_path = os.path.join(complete_path, i)
                    with open(file_path, 'rb') as file:
                        self.train_instances.append(pickle.load(file))
        random.shuffle(self.train_instances)
        print(f"End of loading {len(self.train_instances)} instances!")

    def load_test_instances(self):
        for size in SOLVING_SIZES:
            complete_path = self.base_path+'/test/'+size+'/'
            for i in os.listdir(complete_path):
                if i.endswith('.pkl'):
                    file_path = os.path.join(complete_path, i)
                    with open(file_path, 'rb') as file:
                        self.test_instances.append(pickle.load(file))
        print(f"End of loading {len(self.test_instances)} instances!")