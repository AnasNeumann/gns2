INTERACTIVE = True # Do we want to vizualize the loss function in real time?
SOLVING_REPETITIONS = 10  # solving repetition in inference mode

EPS_START       = 0.99  # starting value of epsilon
EPS_END         = 0.005 # final value of epsilon
EPS_DECAY_RATE  = 0.33  # controls the rate of exponential decay of epsilon, higher means a slower decay (≈35%)
MEMORY_CAPACITY = 60000 # number of transitions in the replay memory
GAMMA           = 1.0   # discount factor
TAU             = 0.003 # update rate of the target network
LEARNING_RATE   = 1e-3  # learning rate for the policy model
BATCH_SIZE      = 64    # batch size at training time
EPISODES        = 10000 # number of training episodes
SWITCH_INSTANCE = 100   # number of episodes before switching instance
SAVING_ITRS     = 100   # number of episodes before saving agents: model weights, optimizers, replay memory, and losses (for security)
MAX_GRAD_NORM   = 2.0   # gradient normalization for really deep networks

YES = 1.0 # feature value for YES 
NO  = 0.0 # feature value for NO

OUTSOURCING   = 0
SCHEDULING    = 1
MATERIAL_USE  = 2
ACTIONS_NAMES = ["outsourcing", "scheduling", "material_use"]
ACTIONS_COLOR = ["blue", "red", "green"]

RM_EMBEDDING_SIZE    = 9   # embedding size for resources and material nodes
OI_EMBEDDING_SIZE    = 16  # embedding size for items and operations nodes
STACK_SIZE           = 2   # stack size of GNN embedding layers
EMBEDDING_HIDDEN_DIM = 64  # hidden dimension while embedding
AGENT_HIDDEN_DIM     = 128 # hidden dimension inside agents' MLPs

W_FINAL  = 0.75 # weight final values versus by-step change in reward computation 
STD_RATE = 0.1  # standardization rate in reward computation

TRAINSET = 0
TESTSET  = 1
TRAINING_SIZES = [['s'], ['s', 'm'], ['s', 'm', 'l'], ['s', 'm', 'l', 'xl'], ['s', 'm', 'l', 'xl', 'xxl']] # sizes at each training stage (curriculum learning)
SOLVING_SIZES  = ['s'] # problem to solve in inference mode