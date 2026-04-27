import torch
import numpy as np
import sys
sys.path.insert(0, 'src')
from analysis.mamba_attn_viz import extract_visual_attention, load_byol_bimodal_encoders, load_etth1, _GradSaliency, _ActHook

# Only use CPU to avoid OOM safely on login node
device = torch.device('cpu')
ts_enc, rp_enc = load_byol_bimodal_encoders("checkpoints/byol_bimodal_full/ts_byol_bimodal_full_lotsa_20260414_171343", device)
ts_np, x = load_etth1("ICML_datasets/ETT-small/ETTh1.csv", 336, 0)
x = x[:, 0:1, :].to(device).float()  # (1, 1, 336)

rp_encoder = rp_enc.eval()
core = rp_encoder.encoder

# Compare core.norm vs core.blocks[-1]
for target_name, target_module in [("LayerNorm", core.norm), ("Last MambaBlock", core.blocks[-1])]:
    print(f"\n--- Hooking {target_name} ---")
    hook_gs = _GradSaliency(target_module)
    hook_act = _ActHook()
    handle_act = target_module.register_forward_hook(hook_act)

    x_req = x.clone().requires_grad_(True)
    out = rp_encoder(x_req)
    score = out.pow(2).sum()
    rp_encoder.zero_grad()
    score.backward()
    handle_act.remove()

    grads = hook_gs._grads.clone().detach().cpu()
    acts = hook_act.output.clone().detach().cpu()

    sal = (grads * acts).abs().sum(dim=-1).mean(dim=0)
    act_norm = acts.norm(dim=-1).mean(dim=0)
    
    print(f"Sal min/max: {sal.min().item():.4f} - {sal.max().item():.4f}")
    print(f"Act norm min/max: {act_norm.min().item():.4f} - {act_norm.max().item():.4f}")
    print(f"Act norm std: {act_norm.std().item():.4f} ({act_norm.std().item()/act_norm.mean().item()*100:.2f}%)")

