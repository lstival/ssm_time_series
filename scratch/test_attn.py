import torch
import numpy as np
import sys
sys.path.insert(0, 'src')
from analysis.mamba_attn_viz import extract_visual_attention, load_byol_bimodal_encoders, _GradSaliency, _ActHook, load_etth1

device = torch.device('cuda')
ts_enc, rp_enc = load_byol_bimodal_encoders("checkpoints/byol_bimodal_full/ts_byol_bimodal_full_lotsa_20260414_171343", device)
ts_np, x = load_etth1("ICML_datasets/ETT-small/ETTh1.csv", 336, 0)

rp_encoder = rp_enc.eval()
x = x.to(device).float().unsqueeze(0)
core = rp_encoder.encoder

hook_gs = _GradSaliency(core.norm)
hook_act = _ActHook()
handle_act = core.norm.register_forward_hook(hook_act)

x_req = x.clone().requires_grad_(True)
out_gs = rp_encoder(x_req)
score = out_gs.pow(2).sum()
rp_encoder.zero_grad()
score.backward()

handle_act.remove()

print("Grads shape:", hook_gs._grads.shape)
print("Acts shape:", hook_gs._acts.shape)
imp = hook_gs.importance().numpy()
print("Grad importance (hook_gs.importance()):")
print(imp)

print("\nMean pooling of activations:")
acts_mean = hook_act.output.norm(dim=-1).float().cpu().numpy()
print(acts_mean)

lags, lag_imp, rp_imp_map = extract_visual_attention(rp_encoder, x, device)
print("Final combined lag_imp:", lag_imp)

