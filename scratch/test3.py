import torch
import numpy as np
import sys
sys.path.insert(0, 'src')
from analysis.mamba_attn_viz import extract_visual_attention, load_byol_bimodal_encoders, load_etth1, _GradSaliency, _ActHook

device = torch.device('cuda')
ts_enc, rp_enc = load_byol_bimodal_encoders("checkpoints/byol_bimodal_full/ts_byol_bimodal_full_lotsa_20260414_171343", device)
ts_np, x = load_etth1("ICML_datasets/ETT-small/ETTh1.csv", 336, 0)
x = x[:, 0:1, :].to(device).float()  # (1, 1, 336)

rp_encoder = rp_enc.eval()
core = rp_encoder.encoder

hook = _GradSaliency(core.norm)
hook_act = _ActHook()
handle_act = core.norm.register_forward_hook(hook_act)

x_req = x.clone().requires_grad_(True)
out = rp_encoder(x_req)
score = out.pow(2).sum()
rp_encoder.zero_grad()
score.backward()
handle_act.remove()

grads = hook._grads.clone().detach().cpu()
acts = hook._acts.clone().detach().cpu()

print("Grads shape:", grads.shape)
# Print variance of grads over the lag dimension (dim=1)
# grads shape: (N, L-1, d_model)
std_over_seq = grads.std(dim=1)
print("Max std of grads over the lag token dimension:", std_over_seq.max().item())

sal = (grads * acts).abs().sum(dim=-1)
print("Saliency values:", sal[0, :10])

# Now with pure activation norm
act_norm = acts.norm(dim=-1)
print("Activation norms:", act_norm[0, :10])
for i in range(acts.shape[0]):
    r = np.corrcoef(sal[i].numpy(), act_norm[i].numpy())[0,1]
    print(f"  Patch {i} correlation:", r)

