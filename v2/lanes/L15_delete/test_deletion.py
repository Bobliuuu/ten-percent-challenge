"""Unit tests for the L15 deletion carrier (CPU, tiny random Qwen3; no download):  python -m pytest v2/lanes/L15_delete/test_deletion.py -q"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
import deletion as DL

def tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    return Qwen3ForCausalLM(cfg).eval()

def test_masked_linear_grad_matches_autograd():
    torch.manual_seed(1); W = torch.randn(6, 5, dtype=torch.float64); x = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    g = (torch.randn(30, dtype=torch.float64) * 0.01).requires_grad_(True); s = 100.0
    y = DL._MaskedLinear.apply(x, g, W, s); gy = torch.randn_like(y); (y * gy).sum().backward(); gx1, gg1 = x.grad.clone(), g.grad.clone()
    x.grad = None; g.grad = None; y2 = F.linear(x, W * torch.sigmoid(g.view(6, 5) * s)); (y2 * gy).sum().backward()
    assert torch.allclose(y, y2) and torch.allclose(gx1, x.grad) and torch.allclose(gg1, g.grad)

def test_save_load_apply_undo_exact():
    m = tiny(); w0 = {i: l.mlp.gate_proj.weight.detach().clone() for i, l in enumerate(m.model.layers)}
    D = {"layers": [0, 1], "target": "gate_proj", "shapes": [[64, 32]] * 2, "idx": [torch.tensor([0, 5, 2047]), torch.tensor([7])]}
    f = tempfile.NamedTemporaryFile(delete=False); f.close(); DL.save(D, f.name); D2 = DL.load(f.name); os.remove(f.name)
    assert all(torch.equal(a, b) for a, b in zip(D["idx"], D2["idx"]))
    sv = DL.apply(m, D2); w = m.model.layers[0].mlp.gate_proj.weight.view(-1)
    assert float(w[0]) == float(w[5]) == float(w[2047]) == 0.0 and (m.model.layers[0].mlp.gate_proj.weight != w0[0]).sum() <= 3
    DL.undo(m, sv); assert all(torch.equal(l.mlp.gate_proj.weight, w0[i]) for i, l in enumerate(m.model.layers))

def test_learn_leaves_model_unchanged_and_only_deletes():
    m = tiny(); sd0 = {k: v.clone() for k, v in m.state_dict().items()}
    seqs = [([1, 2, 3, 4, 5, 6], [4, 5], [9, 10]), ([3, 2, 1, 7], [2, 3], [9, 10])]
    D = DL.learn_deletions(m, None, None, None, None, dict(steps=30, beta=0, bs=2, lr=5e-3, init_sparsity=0.01), seqs=seqs)
    assert all(torch.equal(v, m.state_dict()[k]) for k, v in sd0.items())                      # weights untouched, forward restored
    assert D["stats"]["n_deleted"] > 0 and all(i.max() < 64 * 32 for i in D["idx"] if len(i))
    ids = torch.tensor([seqs[0][0]]); before = m(ids).logits; sv = DL.apply(m, D); after = m(ids).logits; DL.undo(m, sv)
    assert not torch.equal(before, after) and torch.equal(before, m(ids).logits)
    # the trained mask lowers the annotated-target loss vs the original model
    ce = lambda: DL._ce(m, seqs, m.device).item(); c0 = ce(); sv = DL.apply(m, D); c1 = ce(); DL.undo(m, sv)
    assert c1 < c0, (c0, c1)
