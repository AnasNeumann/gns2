LAST_SUCCESS        = 900
SOLVING_REPETITIONS = 10

EPS_START       = 0.99  # starting value of epsilon
EPS_END         = 0.005 # final value of epsilon
EPS_DECAY_RATE  = 0.33  # controls the rate of exponential decay of epsilon, higher means a slower decay (≈35%)
MEMORY_CAPACITY = 60000 # number of transitions in the replay memory
GAMMA           = 1.0   # discount factor
TAU             = 0.003 # update rate of the target network
LEARNING_RATE   = 1e-3
BATCH_SIZE      = 128

YES = 1.0
NO  = 0.0

OUTSOURCING   = 0
SCHEDULING    = 1
MATERIAL_USE  = 2
ACTIONS_NAMES = ["outsourcing", "scheduling", "material_use"]

RM_EMBEDDING_SIZE    = 9
OI_EMBEDDING_SIZE    = 16
STACK_SIZE           = 2
EMBEDDING_HIDDEN_DIM = 64
AGENT_HIDDEN_DIM     = 128