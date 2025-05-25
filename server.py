from concurrent.futures import ThreadPoolExecutor

import litserve as ls
import numpy as np
import torch
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from utils import batch_smiles, linear_model_red_only, predictions, prepare_batch

max_length = 54

VAULT_URL = "https://rodc-kv.vault.azure.net/"
SECRET_NAME = "animakernel"


credential = DefaultAzureCredential()
client = SecretClient(vault_url=VAULT_URL, credential=credential)
SECRET_KEY = client.get_secret(SECRET_NAME).value


class Kernel(ls.LitAPI):
    _device = "cpu"

    def authorize(self, auth: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        if auth.scheme != "Bearer" or auth.credentials != SECRET_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")

    def setup(self, device):
        self.red = torch.jit.load("model_scripted_red.pt").eval().to(device)
        self.ox = torch.jit.load("model_scripted_ox.pt").eval().to(device)
        self.pool = ThreadPoolExecutor(2)

    def decode_request(self, request, **kwargs):
        smiles = request["smiles"]
        return smiles

    def batch(self, inputs):
        batched_smiles = list(self.pool.map(batch_smiles, inputs))
        return prepare_batch(batched_smiles)

    def predict(self, x, **kwargs):
        ox = predictions(self.ox, x, self.device)
        red = predictions(self.red, x, self.device)
        return np.reshape([ox, red], (2, -1)).T

    def unbatch(self, output):
        return output.tolist()

    def encode_response(self, output, **kwargs):
        volts = linear_model_red_only(output[1])
        return {"ox/red": output, "voltages": volts}


if __name__ == "__main__":
    api = Kernel(batch_timeout=1, max_batch_size=3)
    server = ls.LitServer(api, workers_per_device=2, accelerator="auto", devices=1)
    server.run(port=8000)
