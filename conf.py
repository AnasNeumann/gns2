LAST_SUCCESS: int = 900

YES = 1.0
NO = 0.0

OUTSOURCING = 0
SCHEDULING = 1
MATERIAL_USE = 2
ACTIONS_NAMES = ["outsourcing", "scheduling", "material_use"]

LEARNING_RATE = 1e-3
SOLVING_REPETITIONS = 10
GNN_CONF = {
    'resource_and_material_embedding_size': 8,
    'operation_and_item_embedding_size': 16,
    'nb_layers': 2,
    'embedding_hidden_channels': 64,
    'actor_hidden_channels': 128}