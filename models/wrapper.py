import torch.nn as nn
from models.tera import tera

def get_model(name):
    if 'tera' in name.lower():
        model = tera(name, init_model=True)
    return model