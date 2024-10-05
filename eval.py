from pathlib import Path
import torch
import torchaudio
import yaml

from models.arch.OnlineSpatialNet import OnlineSpatialNet
from models.io.norm import Norm
from models.io.stft import STFT


def forward(x):
    X, stft_paras = stft.stft(x)  # [B,C,F,T], complex
    B, C, F, T = X.shape
    X, (Xr, XrMM) = norm.norm(X, ref_channel=0)
    X = X.permute(0, 2, 3, 1)  # B,F,T,C; complex
    X = torch.view_as_real(X).reshape(B, F, T, -1)  # B,F,T,2C

    # network process
    out = model.forward(X)
    if not torch.is_complex(out):
        out = torch.view_as_complex(out.float().reshape(B, F, T, -1, 2))  # [B,F,T,Spk]
    out = out.permute(0, 3, 1, 2)  # [B,Spk,F,T]

    yr_hat = stft.istft(out, stft_paras)
    return yr_hat.squeeze()
    ...


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    config_path = "logs/OnlineSpatialNet/version_0/config.yaml"
    ckpt_path = "logs/OnlineSpatialNet/version_0/checkpoints/last.ckpt"
    idx = len("arch.")
    device = "cuda"

    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)
    model = OnlineSpatialNet(**config["model"]["arch"]["init_args"])
    model = torch.compile(model)

    # load weights from checkpoint
    ckpt_dict = torch.load(ckpt_path, map_location=device)
    tar_keys = model.state_dict().keys()
    state_dict = {key[idx:]: value for key, value in ckpt_dict["state_dict"].items() if key[idx:] in tar_keys}

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # inference an audio

    stft: STFT = STFT(n_fft=512, n_hop=256, win_len=512).to(device)
    norm: Norm = Norm(mode="utterance").to(device)

    out_dir = Path("/home/featurize/output/out_wav")
    out_dir.mkdir(parents=True, exist_ok=True)
    for mix_path in Path(r"/home/featurize/data/audio_test/test_mixture").glob("*_mix.wav"):
        tar_path = mix_path.with_name(mix_path.name.replace("mix", "tar"))
        out_path = out_dir / mix_path.name.replace("mix", "out")
        
        mix_data, sr = torchaudio.load(mix_path)
        tar_data, sr = torchaudio.load(tar_path)

        inp_data = mix_data[None].to(device)
        out_data = forward(inp_data).cpu()

        save_data = torch.cat([mix_data[0:1], tar_data, out_data])
        torchaudio.save(out_path, save_data, sr)
        print(out_path)
        ...
