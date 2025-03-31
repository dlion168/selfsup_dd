# upstream/mockingjay.py
import torch
import torch.nn as nn
import s3prl.hub as hub
from s3prl.util.download import _urls_to_filepaths

class tera(nn.Module):
    def __init__(self, name="tera_100hr", init_model = False, layers=[-1]):
        super(tera, self).__init__()
        if init_model :
            # Random initialize model
            self.model = getattr(hub, name.split("_")[0]+"_local")(ckpt="/home/ycevan/.cache/s3prl/download/d896f039ee7a23dbaa50ee488b2b2070c4f6af3eed9039d8e6150541f977c179.states-200000.ckpt?dl=1", options_config="/home/ycevan/selfsup_dd/models/tera_option.yaml")
        else: 
            self.model = getattr(hub, name)()
        self.layers = layers

    def forward(self, wavs):
        """
        參數 wavs 可接受：
          - list of torch.Tensor，每個 tensor 形狀為 (samples,)
          - 或 batch tensor (batch_size, samples)；此時會拆分成 list
        """
        # 若輸入為 batch tensor，拆分成 list
        if isinstance(wavs, torch.Tensor):
            wavs = [wavs[i] for i in range(wavs.shape[0])]

        output = self.model(wavs)
        
        return torch.stack([output["hidden_states"][l] for l in self.layers], dim=0)
