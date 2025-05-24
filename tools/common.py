import resource
from typing import Union, Any, Dict

# #####################################################################
# =*= COMMON TOOLS (objects and functions) USED ACCROSS THE PROJECT =*=
# #####################################################################
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

class ProjectDirectory:
    def __init__(self, data: str, instances: str, models: str, results: str, out: str, scripts: str, solutions: str):
        self.data: str = data
        self.instances: str = self.data+'/'+instances
        self.models: str = self.data+'/'+models
        self.out: str = self.data+'/'+out
        self.results: str = self.data+'/'+results
        self.solutions: str = self.data+'/'+solutions
        self.scripts: str = scripts

directory = ProjectDirectory('data', 'instances', 'models', 'results', 'out', 'jobs/scripts', 'solutions')

num_feature = Union[int, float]
all_types_feature = Union[int, float, bool, list]
generic_object = Union[object, Dict[Any, Any]]

def set_memory_limit(max_ram_bytes: num_feature):
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (max_ram_bytes * 1024 * 1024 * 1024, hard))

def init_1D(a: int, default_value: all_types_feature):
    return [default_value] * a

def init_2D(a: int, b: int, default_value: all_types_feature):
    return [[default_value for _ in range(b)] for _ in range(a)]

def init_3D(a: int, b: int, c: int, default_value: all_types_feature):
    return [[[default_value for _ in range(c)] for _ in range(b)] for _ in range(a)]

def init_several_1D(a: int, default_value: all_types_feature, nb: int):
    return (init_1D(a, default_value) for _ in range(nb))

def init_several_2D(a: int, b: int, default_value: all_types_feature, nb: int):
    return (init_2D(a, b, default_value) for _ in range(nb))

def init_several_3D(a: int, b: int, c: int, default_value: all_types_feature, nb: int):
    return (init_3D(a, b, c, default_value) for _ in range(nb))

def to_bool(s: str):
    return s.lower() in ['true', '1', 't', 'y', 'yes']

def search_object_by_id(objects: list[generic_object], id: int):
    for obj in objects:
        if obj['id'] == id:
            return obj
    return None

def objective_value(cmax: int, cost: int, cmax_weight: float) -> int:
    cmax_weight = int(100 * cmax_weight)
    cost_weight = 100 - cmax_weight
    return cmax*cmax_weight + cost*cost_weight

def debug_printer(mode):
    if mode:
        def debug_print(*args):
            print(*args)
            with open('./log.sh', 'a') as file:
                file.write(*args)
                file.write('\n')
    else:
        def debug_print(*_):
            pass
    return debug_print