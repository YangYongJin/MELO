from audioop import bias
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models.meta_model import MetaBERT4Rec
from models.meta_model import MetaLossNetwork
from options import args
import numpy as np


def update_params(names_weights_dict, names_grads_wrt_params_dict):
    """Applies a single gradient descent update to all parameters.
    All parameter updates are performed using in-place operations and so
    nothing is returned.
    Args:
        grads_wrt_params: A list of gradients of the scalar loss function
            with respect to each of the parameters passed to `initialise`
            previously, with this list expected to be in the same order.
    """
    return {
        key: names_weights_dict[key]
        - 1e-3 * names_grads_wrt_params_dict[key]
        for key in names_weights_dict.keys()
    }


def get_inner_loop_parameter_dict(params):
    """
    Returns a dictionary with the parameters to use for inner loop updates.
    :param params: A dictionary of the network's parameters.
    :return: A dictionary of the parameters to use for the inner loop optimization process.
    """
    return {
        name: param
        for name, param in params
        if param.requires_grad
    }


model = MetaBERT4Rec(
    args)

loss_net = MetaLossNetwork()

names_weights_copy = get_inner_loop_parameter_dict(
    model.named_parameters())

loss_weights_copy = get_inner_loop_parameter_dict(
    loss_net.named_parameters())

# print(names_weights_copy.keys())
# print(names_weights_copy['bert.bert_embedding.embedding.weights'].shape)
# # 1/0

model.zero_grad()

# for p in model.parameters():
#     print(p)

inputs = (torch.LongTensor(np.ones((32, 1))), torch.LongTensor(np.ones((32, 29))),
          torch.LongTensor(np.ones((32, 1))), torch.FloatTensor(np.ones((32, 29))))
input2 = (torch.LongTensor(np.ones((32, 1))), torch.LongTensor(np.ones((32, 29))),
          torch.LongTensor(np.ones((32, 1))), torch.FloatTensor(np.ones((32, 29))))
output = torch.randn(32, 1)
output2 = torch.randn(32, 1)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
a = torch.tensor(4.2)
a.requires_grad = True
b = torch.tensor(4.2)
b.requires_grad = True

loss_net = nn.Linear(1, 1, bias=False)
loss_net.zero_grad()

for _ in range(5):
    # names_weights_copy.zero_grad()
    loss = loss_fn(model(inputs, params=names_weights_copy), output)
    support_task_state = []
    support_task_state.append(loss)
    # for v in names_weights_copy.values():
    #     support_task_state.append(v.mean())
    support_task_state = torch.stack(support_task_state)
    loss2 = torch.mean(
        loss_net(torch.mean(support_task_state).reshape(1, 1)))
    # loss2 = torch.mean(support_task_state)*a

    grads = torch.autograd.grad(
        loss2, names_weights_copy.values(), allow_unused=True, create_graph=True)

    names_grads_copy = dict(zip(names_weights_copy.keys(), grads))
    names_weights_copy = update_params(names_weights_copy, names_grads_copy)

print(a.grad)
loss = loss_fn(model(inputs, params=names_weights_copy), output)
loss.backward()
# print(a.grad)

# total_norm = 0.0
# for p in model.parameters():
#     param_norm = p.grad.detach().data.norm(2)
#     total_norm += param_norm.item() ** 2
# total_norm = total_norm ** 0.5
# print(total_norm)

total_norm = 0.0
for p in loss_net.parameters():
    param_norm = p.grad.detach().data.norm(2)
    total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5

print(total_norm)
print(a)
