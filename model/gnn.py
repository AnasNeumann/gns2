import torch
from torch.nn import Sequential, Linear, ReLU, Module, ModuleList, LayerNorm, Dropout, Identity
from torch import Tensor
from model.graph import State
from torch_geometric.nn import GATConv, AttentionalAggregation
from model.graph import FC as f
from conf import *
import torch.nn.functional as F

# ##########################################################
# =*= GRAPH ATTENTION NEURAL NETWORK (GNN): ARCHITECTURE =*=
# ##########################################################
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"

def dim(v):
    return len(v)

class ResidualMLP(Module):
    def __init__(self, in_dim:  int, out_dim: int, dropout: float=DROPOUT, activation=F.relu):
        super().__init__()
        self.activation = activation
        self.fc1 = Linear(in_dim, out_dim)
        self.fc2 = Linear(out_dim, out_dim)
        self.dropout = Dropout(dropout)
        self.res_proj = (Identity() if in_dim == out_dim else Linear(in_dim, out_dim))
        self.norm = LayerNorm(out_dim)

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.norm(self.res_proj(x) + out)
        return out

class EmbbedingGNN(Module):
    def __init__(self, d_model: int = D_MODEL, stack: int = STACK, num_heads: int = HEADS, dropout: float = DROPOUT):
        super(EmbbedingGNN, self).__init__()
        self.stack                          = stack
        self.d_model                        = d_model

        self.material_upscale               = Linear(dim(f.material), d_model)
        self.resource_upscale               = Linear(dim(f.resource), d_model)
        self.item_upscale                   = Linear(dim(f.item), d_model)
        self.operation_upscale              = Linear(dim(f.operation), d_model)
        self.need_for_materials_upscale     = Linear(dim(f.need_for_materials), d_model)
        self.need_for_resources_upscale     = Linear(dim(f.need_for_resources), d_model)
        self.rev_need_for_materials_upscale = Linear(dim(f.need_for_materials), d_model)
        self.rev_need_for_resources_upscale = Linear(dim(f.need_for_resources), d_model)

        self.GAT_stack_ops_for_mat          = ModuleList() # 'operation', '->', 'material'
        self.GAT_stack_ops_for_res          = ModuleList() # 'operation', '->', 'resource'

        self.GAT_stack_ops_for_item         = ModuleList() # 'operation', '->', 'item'
        self.GAT_stack_parents              = ModuleList() # 'parent item', '->', 'children item'
        self.GAT_stack_children             = ModuleList() # 'children item', '->', 'parent item'

        self.GAT_stack_item_for_op          = ModuleList() # 'item', '->', 'operation'
        self.GAT_stack_preds                = ModuleList() # 'pred operation', '->', 'succ operation'
        self.GAT_stack_succs                = ModuleList() # 'succ operation', '->', 'pred operation'
        self.GAT_stack_res_for_op           = ModuleList() # 'resource', '->', 'operation'
        self.GAT_stack_mat_for_op           = ModuleList() # 'material', '->', 'operation'
  
        for _ in range(stack):
            self.GAT_stack_ops_for_mat.append(GATConv(in_channels=(d_model, d_model), out_channels=d_model // num_heads, heads=num_heads, concat=True, edge_dim=d_model, add_self_loops=False)) # 'operation', '->', 'material'
            self.GAT_stack_ops_for_res.append(GATConv(in_channels=(d_model, d_model), out_channels=d_model // num_heads, heads=num_heads, concat=True, edge_dim=d_model, add_self_loops=False)) # 'operation', '->', 'resource'
            
            self.GAT_stack_ops_for_item.append(GATConv(in_channels=(d_model, d_model), out_channels=d_model // num_heads, heads=num_heads, concat=True, add_self_loops=False)) # 'operation', '->', 'item'
            self.GAT_stack_parents.append(GATConv(in_channels=(d_model, d_model), out_channels=d_model // num_heads, heads=num_heads, concat=True, add_self_loops=False)) # 'parent item', '->', 'children item'
            self.GAT_stack_children.append(GATConv(in_channels=(d_model, d_model), out_channels=d_model // num_heads, heads=num_heads, concat=True, add_self_loops=False)) # 'children item', '->', 'parent item'
            
            self.GAT_stack_item_for_op.append(GATConv(in_channels=(d_model, d_model), out_channels=d_model // num_heads, heads=num_heads, concat=True, add_self_loops=False)) # 'item', '->', 'operation'
            self.GAT_stack_preds.append(GATConv(in_channels=(d_model, d_model), out_channels=d_model // num_heads, heads=num_heads, concat=True, add_self_loops=False)) # 'pred operation', '->', 'succ operation'
            self.GAT_stack_succs.append(GATConv(in_channels=(d_model, d_model), out_channels=d_model // num_heads, heads=num_heads, concat=True, add_self_loops=False)) # 'succ operation', '->', 'pred operation'
            self.GAT_stack_res_for_op.append(GATConv(in_channels=(d_model, d_model), out_channels=d_model // num_heads, heads=num_heads, concat=True, edge_dim=d_model, add_self_loops=False)) # 'resource', '->', 'operation'
            self.GAT_stack_mat_for_op.append(GATConv(in_channels=(d_model, d_model), out_channels=d_model // num_heads, heads=num_heads, concat=True, edge_dim=d_model, add_self_loops=False)) # 'material', '->', 'operation'

        self.material_MLP               = ResidualMLP(2 * d_model, d_model, dropout=dropout)
        self.resource_MLP               = ResidualMLP(2 * d_model, d_model, dropout=dropout)
        self.item_MLP                   = ResidualMLP(4 * d_model, d_model, dropout=dropout)
        self.operation_MLP              = ResidualMLP(6 * d_model, d_model, dropout=dropout)
    
    def forward(self, state: State):
        x_operations     = self.operation_upscale(state.operations)
        x_items          = self.item_upscale(state.items)
        x_materials      = self.material_upscale(state.materials)
        x_resources      = self.resource_upscale(state.resources)
        x_material_needs = self.need_for_materials_upscale(state.need_for_materials.edge_attr)
        x_resource_needs = self.need_for_resources_upscale(state.need_for_resources.edge_attr)
        x_mat_for_ops    = self.rev_need_for_materials_upscale(state.rev_need_for_materials.edge_attr)
        x_res_for_ops    = self.rev_need_for_resources_upscale(state.rev_need_for_resources.edge_attr)

        for l in range(self.stack):
            ops_for_mat  = self.GAT_stack_ops_for_mat[l]((x_operations, x_materials), state.need_for_materials.edge_index, size=(x_operations.size(0), x_materials.size(0)), edge_attr=x_material_needs)  # 'operation', '->', 'material'
            ops_for_res  = self.GAT_stack_ops_for_res[l]((x_operations, x_resources), state.need_for_resources.edge_index, size=(x_operations.size(0), x_resources.size(0)), edge_attr=x_resource_needs)  # 'operation', '->', 'resource'
            
            ops_for_item = self.GAT_stack_ops_for_item[l]((x_operations, x_items), state.operations_of_item.edge_index, size=(x_operations.size(0), x_items.size(0)))  # 'operation', '->', 'item'
            parents      = self.GAT_stack_parents[l]((x_items, x_items), state.parent_assembly.edge_index, size=(x_items.size(0), x_items.size(0))) # 'parent item', '->', 'children item'
            children     = self.GAT_stack_children[l]((x_items, x_items), state.children_assembly.edge_index, size=(x_items.size(0), x_items.size(0)))  # 'children item', '->', 'parent item'

            item_for_op  = self.GAT_stack_item_for_op[l]((x_items, x_operations), state.operation_assembly.edge_index, size=(x_items.size(0), x_operations.size(0))) # 'item', '->', 'operation'
            preds        = self.GAT_stack_preds[l]((x_operations, x_operations), state.precedences.edge_index, size=(x_operations.size(0), x_operations.size(0))) # 'prec operation', '->', 'succ operation'
            succs        = self.GAT_stack_succs[l]((x_operations, x_operations), state.successors.edge_index, size=(x_operations.size(0), x_operations.size(0))) # 'succ operation' '->', 'prec operation'
            res_for_op   = self.GAT_stack_res_for_op[l]((x_resources, x_operations), state.rev_need_for_resources.edge_index, size=(x_resources.size(0), x_operations.size(0)), edge_attr=x_res_for_ops) # 'resource', '->', 'operation'
            mat_for_op   = self.GAT_stack_mat_for_op[l]((x_materials, x_operations), state.rev_need_for_materials.edge_index, size=(x_materials.size(0), x_operations.size(0)), edge_attr=x_mat_for_ops) # 'material', '->', 'operation'

            x_materials  = self.material_MLP(torch.cat([x_materials, ops_for_mat], dim=-1))
            x_resources  = self.resource_MLP(torch.cat([x_resources, ops_for_res], dim=-1))
            x_items      = self.item_MLP(torch.cat([x_items, ops_for_item, parents, children], dim=-1))
            x_operations = self.operation_MLP(torch.cat([x_operations, preds, succs, item_for_op, res_for_op, mat_for_op], dim=-1))
        return x_materials, x_resources, x_items, x_operations 

class SchedulingActor(Module):
    def __init__(self, d_model: int=D_MODEL, actor_dim: int=ACTOR_DIM):
        super(SchedulingActor, self).__init__()
        self.gnn               = EmbbedingGNN()
        first_dimension        = actor_dim
        second_dimenstion      = int(actor_dim / 2)
        self.resource_pooling  = AttentionalAggregation(Linear(d_model, 1))
        self.item_pooling      = AttentionalAggregation(Linear(d_model, 1))
        self.operation_pooling = AttentionalAggregation(Linear(d_model, 1))
        self.actor = Sequential(
            Linear(5 * d_model + 1, first_dimension), ReLU(),
            Linear(first_dimension, second_dimenstion), ReLU(),
            Linear(second_dimenstion, 1)
        )

    def forward(self, state: State, actions: list[(int, int)], alpha: Tensor):
        _, r_embbedings, i_embeddings, o_embeddings = self.gnn(state)
        pooled_resources  = self.resource_pooling(r_embbedings)
        pooled_items      = self.item_pooling(i_embeddings, index=torch.zeros_like(i_embeddings[:,0], dtype=torch.long))
        pooled_operations = self.operation_pooling(o_embeddings)
        state_embedding   = torch.cat([torch.cat([pooled_items, pooled_operations, pooled_resources], dim=-1)[0], alpha], dim=0).unsqueeze(0).expand(len(actions), -1)
        operations_ids, resources_ids = zip(*actions)
        inputs            = torch.cat([o_embeddings[list(operations_ids)], r_embbedings[list(resources_ids)], state_embedding], dim=1) # shape = [possible decision, concat embedding]
        return self.actor(inputs)

class OutousrcingActor(Module):
    def __init__(self, d_model: int=D_MODEL, actor_dim: int=ACTOR_DIM):
        super(OutousrcingActor, self).__init__()
        self.gnn               = EmbbedingGNN()
        first_dimension        = actor_dim
        second_dimenstion      = int(actor_dim / 2)
        self.item_pooling      = AttentionalAggregation(Linear(d_model, 1))
        self.operation_pooling = AttentionalAggregation(Linear(d_model, 1))
        self.actor = Sequential(
            Linear(3 * d_model + 2, first_dimension), ReLU(),
            Linear(first_dimension, second_dimenstion), ReLU(),
            Linear(second_dimenstion, 1)
        )

    def forward(self, state: State, actions: list[(int, int)], alpha: Tensor):
        _, _, i_embeddings, o_embeddings = self.gnn(state)
        pooled_items      = self.item_pooling(i_embeddings, index=torch.zeros_like(i_embeddings[:,0], dtype=torch.long))
        pooled_operations = self.operation_pooling(o_embeddings)
        state_embedding   = torch.cat([torch.cat([pooled_items, pooled_operations], dim=-1)[0], alpha], dim=0).unsqueeze(0).expand(len(actions), -1)
        item_ids, outsourcing_choices = zip(*actions)
        outsourcing_choices_tensor = torch.tensor(outsourcing_choices, dtype=torch.float32, device=alpha.device).unsqueeze(1)
        inputs            = torch.cat([i_embeddings[list(item_ids)], outsourcing_choices_tensor, state_embedding], dim=1)
        return self.actor(inputs)
    
class MaterialActor(Module):
    def __init__(self, d_model: int=D_MODEL, actor_dim: int=ACTOR_DIM):
        super(MaterialActor, self).__init__()
        self.gnn               = EmbbedingGNN()
        first_dimension        = actor_dim
        second_dimenstion      = int(actor_dim / 2)
        self.material_pooling  = AttentionalAggregation(Linear(d_model, 1))
        self.item_pooling      = AttentionalAggregation(Linear(d_model, 1))
        self.operation_pooling = AttentionalAggregation(Linear(d_model, 1))
        self.actor = Sequential(
            Linear(5 * d_model + 1, first_dimension), ReLU(),
            Linear(first_dimension, second_dimenstion), ReLU(),
            Linear(second_dimenstion, 1)
        )

    def forward(self, state: State, actions: list[(int, int)], alpha: Tensor):
        m_embeddings, _, i_embeddings, o_embeddings = self.gnn(state)
        pooled_materials  = self.material_pooling(m_embeddings)
        pooled_items      = self.item_pooling(i_embeddings, index=torch.zeros_like(i_embeddings[:,0], dtype=torch.long))
        pooled_operations = self.operation_pooling(o_embeddings)
        state_embedding   = torch.cat([torch.cat([pooled_items, pooled_operations, pooled_materials], dim=-1)[0], alpha], dim=0).unsqueeze(0).expand(len(actions), -1)
        operations_ids, materials_ids = zip(*actions)
        inputs            = torch.cat([o_embeddings[list(operations_ids)], m_embeddings[list(materials_ids)], state_embedding], dim=1)
        return self.actor(inputs)