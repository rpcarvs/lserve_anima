import numpy as np
import torch

max_length = 54


def linear_model_red_only(red):
    return np.multiply(0.8809, red) - 0.7833


def batch_smiles(smiles):
    return torch.tensor(smiles)


def prepare_batch(batched_smiles):
    packing = torch.nn.utils.rnn.pack_sequence(batched_smiles, enforce_sorted=False)
    packing_padding = torch.nn.utils.rnn.pad_packed_sequence(
        packing, batch_first=True, total_length=max_length
    )
    return packing_padding[0][:, :, 0]


def predictions(model, processed_smiles, device):
    with torch.inference_mode():
        inputs = processed_smiles.to(device)
        output = model(inputs)
    return output.cpu().detach().numpy().tolist()
