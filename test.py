from models.base import MetaLinearLayer
import torch
import torch.nn as nn


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


model = MetaLinearLayer(4, 1)
names_weights_copy = {
    name: param
    for name, param in model.named_parameters()
    if param.requires_grad
}

loss_n = nn.Sequential(nn.Linear(2, 1, bias=False),
                       nn.Linear(1, 1, bias=False))
a = nn.Parameter(torch.ones(1, 1))


inp = torch.randn(8, 4)
inp2 = torch.randn(8, 4)
output = torch.randn(8, 1)
loss_fn = nn.MSELoss()

for i in range(5):
    loss = loss_fn(model(inp, params=names_weights_copy), output)
    loss2 = loss.view(1, 1)
    loss3 = loss_n(torch.cat((loss2, a), dim=1))

    grads = torch.autograd.grad(loss3, names_weights_copy.values(),
                                allow_unused=True, create_graph=True)
    names_grads_copy = dict(zip(names_weights_copy.keys(), grads))

    for key, grad in names_grads_copy.items():
        if grad is None:
            print('Grads not found for inner loop parameter', key)
        names_grads_copy[key] = names_grads_copy[key].sum(dim=0)

    names_weights_copy = update_params(
        names_weights_dict=names_weights_copy, names_grads_wrt_params_dict=names_grads_copy)

loss = loss_fn(model(inp2, params=names_weights_copy), output)
loss.backward()

total_norm = 0.0
for p in loss_n.parameters():
    print(p.grad.detach().data)
    param_norm = p.grad.detach().data.norm(2)
    total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5
print(total_norm)

print(a.grad)  # None
