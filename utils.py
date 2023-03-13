import torch
import torch.nn as nn

import random
import numpy as np

import itertools

def parity(n, k, n_samples, seed=42):
    'Data generation'

    random.seed(seed)
    samples = torch.Tensor([[random.choice([-1, 1]) for j in range(n)] for i in range(n_samples)])
    # targets = torch.prod(input[:, n//2:n//2+k], dim=1) # parity hidden in the middle
    targets = torch.prod(samples[:, :k], dim=1) # parity hidden in first k bits

    return samples, targets

@torch.no_grad()
def get_sensitivity(predict_fn, inputs, argmax: bool = True):
    """Get sensitivity of predict_fn on inputs.

    Returns normalized value in [0, 1], where 0 means no sensitivity, and 1 means sensitivity of n_features.

    Args:
        predict_fn: Function to evaluate.
        inputs: Boolean tensor of shape [n_samples, n_features]
        argmax: Reduce output of model_fn on the final axis by taking argmax (for logits).
    """
    _, n_features = inputs.size()
    index = torch.arange(n_features)
    n_flipped = 0
    n_total = 0
    for sample in inputs:
        source = predict_fn(sample)
        batch = sample.repeat(n_features, 1)
        batch[index, index] = -batch[index, index]
        neighbors = predict_fn(batch)
        if argmax:
            source = source.sign()
            neighbors = neighbors.sign()
        n_flipped += (neighbors != source).sum().item()
        n_total += len(neighbors)
    return n_flipped / n_total

def sensitivity_calc(data, model, mask_idx, device='cuda', args=None):
    model.eval()

    if (mask_idx is not None):
        # create mask
        idx = torch.LongTensor([mask_idx for _ in range(data.shape[1])])
        mask = torch.zeros(data.shape[1], args.width)
        mask.scatter_(1, idx, 1.)

        predict_fn = lambda x: model.masked_forward(x, mask.to(device))
    else:
        predict_fn = lambda x: model(x)

    return args.n * get_sensitivity(predict_fn, data.to(device))


def mean_and_std_across_seeds(list_stats):
    array_stats = np.array(list_stats)
    mean = np.average(array_stats, axis=0)
    std =  np.std(array_stats, axis=0)

    return mean, std

def acc_calc(dataloader, model, mask_idx=None, device='cuda', args=None, faithfulness=False):
    model.eval()

    acc, total = 0, 0
    for id, (x_batch, y_batch) in enumerate(dataloader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        if (mask_idx is not None):
            # create mask
            idx = torch.LongTensor([mask_idx for _ in range(len(y_batch))])
            mask = torch.zeros(len(y_batch), args.width)
            mask.scatter_(1, idx, 1.)

            pred = model.masked_forward(x_batch, mask.to(device))
            if (faithfulness):
                fullmodel_pred = model(x_batch)
        else:
            pred = model(x_batch)
        if (faithfulness):
            acc += (torch.sign(torch.squeeze(pred)) == torch.sign(torch.squeeze(fullmodel_pred))).sum().item()    
        else:
            acc += (torch.sign(torch.squeeze(pred)) == y_batch).sum().item()
        total += x_batch.shape[0]
    
    return acc / total


def loss_calc(dataloader, model, loss_fn, device='cuda'):
    model.eval()

    loss, total = 0, 0
    for id, (x_batch, y_batch) in enumerate(dataloader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        pred = model(x_batch)
        loss += loss_fn(pred, y_batch).sum().item()
        total += x_batch.shape[0]

    return loss / total

def circuit_discovery_linear(epoch, saved_model, norms, dataloader, device='cuda', args=None):
    # Calculate least number of neurons that recovers original (train set) performance with linear search

    values = np.array(norms['feats'][epoch]).argsort()
    for k in range(1, args.width):
        idx = values[-k:]
        masked_acc = acc_calc(dataloader, saved_model, idx, device=device, args=args)
        full_acc = acc_calc(dataloader, saved_model, device=device, args=args)

        if (masked_acc == full_acc):
            return k, idx
    
    return float('inf'), None # mistake
    
    
def circuit_discovery_binary(epoch, saved_model, norms, dataloader, device='cuda', args=None):
    # Calculate least number of neurons that recovers original (train set) performance with binary search (assuming that it increases monotonically)
    
    left, right = 1, args.width
    prev_k = -1
    min_k, min_idx = float('inf'), None

    values = np.array(norms['feats'][epoch]).argsort()
    while left < right:
        k = (left + right) // 2
        if (prev_k == k):
            break

        idx = values[-k:]
        masked_acc = acc_calc(dataloader, saved_model, idx, device=device, args=args)
        full_acc = acc_calc(dataloader, saved_model, device=device, args=args)

        if (masked_acc == full_acc) and (k < min_k):
            min_k = k
            min_idx = idx
        if (masked_acc < full_acc):
            left = k
        else:
            right = k + 1

        prev_k = k
    
    return min_k, min_idx

class FF1(torch.nn.Module):
    def __init__(self, input_dim=40, width=1000):
        super(FF1, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, width)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(width, 1, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

    def masked_forward(self, x, mask):
        x = self.linear1(x)
        x = self.activation(x)
        x = x * mask
        x = self.linear2(x)
        return x

class MyHingeLoss(torch.nn.Module):
    def __init__(self):
        super(MyHingeLoss, self).__init__()

    def forward(self, output, target):
        hinge_loss = 1 - torch.mul(torch.squeeze(output), torch.squeeze(target))
        hinge_loss[hinge_loss < 0] = 0
        return hinge_loss

class ArityFinder:
    def __init__(self, model, inputs):
        self.model = model
        self.inputs = inputs
        self.predictions = self.predict()

    def predict(self):
        return self.model(self.inputs).sign()
    
    @torch.no_grad()
    def get_active_inputs(self, key: str = "linear1") -> list:
        self.linear = self.model.get_submodule(key)
        arange = torch.arange(self.linear.weight.size(1)).to('cuda')
        all_active = []
        for neuron, weights in enumerate(self.linear.weight):
            self.neuron = neuron
            self.old_weights = weights.data.clone()
            self.indices = weights.abs().argsort()
            n_pruned = self.get_n_pruned(0, len(weights))
            active = arange[self.indices >= n_pruned]
            all_active.append(active.tolist())

        return all_active

    @torch.no_grad()
    def get_arities(self, key: str = "linear1") -> list:
        self.linear = self.model.get_submodule(key)
        arities = []
        for neuron, weights in enumerate(self.linear.weight):
            self.neuron = neuron
            self.old_weights = weights.data.clone()
            self.indices = weights.abs().argsort()
            n_pruned = self.get_n_pruned(0, len(weights))
            arities.append(len(weights) - n_pruned)
        return arities

    def get_n_pruned(self, floor, ceil) -> int:
        """Get the arity of self.neuron.

        Args:
            floor: Number we know we can prune.
            ceil: Number we could possibly prune.
        
        # TODO: Could possibly simplify base case.
        """
        if floor == ceil:
            return floor
        # elif floor == ceil - 1:
        #     mask = (self.indices >= ceil)
        #     self.linear.weight.data[self.neuron] *= mask
        #     predictions = self.predict()
        #     self.linear.weight.data[self.neuron] = self.old_weights
        #     if (predictions == self.predictions).all():
        #         return ceil
        #     else:
        #         return floor

        midpoint = (floor + ceil + 1) // 2
        mask = (self.indices >= midpoint)
        self.linear.weight.data[self.neuron] *= mask
        predictions = self.predict()
        self.linear.weight.data[self.neuron] = self.old_weights
        if (predictions == self.predictions).all():
            return self.get_n_pruned(midpoint, ceil)
        else:
            return self.get_n_pruned(floor, midpoint - 1)