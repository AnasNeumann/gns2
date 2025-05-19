SOLVING_REPETITIONS = 10  # solving repetition in inference mode

EPS_START       = 0.99   # starting value of epsilon
EPS_END         = 0.005  # final value of epsilon
EPS_DECAY_RATE  = 0.33   # controls the rate of exponential decay of epsilon, higher means a slower decay (≈35%)
MEMORY_CAPACITY = 100000 # number of transitions in the replay memory
GAMMA           = 0.3    # discount factor
DELTA           = 2.0    # Huber loss "threshold" parameter (distinguishing between small and big errors)
TAU             = 0.003  # update rate of the target network
LEARNING_RATE   = 2e-4   # learning rate for the policy model
BATCH_SIZE      = 64     # batch size at training time
EPISODES        = 5000   # number of training episodes
SWITCH_INSTANCE = 100    # number of episodes before switching instance
SAVING_ITRS     = 100    # number of episodes before saving agents: model weights, optimizers, replay memory, and losses (for security)
MAX_GRAD_NORM   = 5.0    # gradient normalization for really deep networks
TRAINING_ITRS   = 5      # number of solving episodes before training the agents!

YES = 1.0 # feature value for YES 
NO  = 0.0 # feature value for NO

OUTSOURCING   = 0
SCHEDULING    = 1
MATERIAL_USE  = 2
ACTIONS_NAMES = ["outsourcing", "scheduling", "material_use"]
ACTIONS_COLOR = ["blue", "red", "green"]

RM_EMBEDDING_SIZE    = 8   # embedding size for resources and material nodes
OI_EMBEDDING_SIZE    = 16  # embedding size for items and operations nodes
STACK_SIZE           = 2   # stack size of GNN embedding layers
EMBEDDING_HIDDEN_DIM = 64  # hidden dimension while embedding
AGENT_HIDDEN_DIM     = 128 # hidden dimension inside agents' MLPs

D_MODEL   = 16
STACK     = 2
DROPOUT   = 0.1
HEADS     = 4
ACTOR_DIM = 64

W_FINAL  = 0.85  # weight final values versus by-step change in reward computation 
STD_RATE = 0.15  # standardization rate in reward computation

TRAINSET = 0
TESTSET  = 1
TRAINING_SIZES = [['s'], ['s', 'm'], ['s', 'm', 'l'], ['s', 'm', 'l', 'xl'], ['s', 'm', 'l', 'xl', 'xxl']] # sizes at each training stage (curriculum learning)
SOLVING_SIZES  = ['s'] # problem to solve in inference mode