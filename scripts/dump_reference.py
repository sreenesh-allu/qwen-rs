#!/usr/bin/env python3
"""
Dump reference activations + logits from HF transformers (PyTorch, fp32, eager attention)
so the Rust engine can be checked layer by layer.

Usage:
    python scripts/dump_reference.py \
        --model models/qwen2.5-0.5b-instruct \
        --out fixtures \
        --revision <hf-commit-sha>        # optional, recorded in manifest for reproducibility

Writes:
    fixtures/manifest.json             prompts, exact rendered text, input_ids, greedy_ids, versions
    fixtures/<prompt_name>.safetensors f32 tensors (batch dim dropped):
        embed                      [seq, 896]          output of embed_tokens
        rope.cos / rope.sin        [seq, 64]           HF's RoPE tables (half-rotation layout)
        l0.input_norm              [seq, 896]          layer 0 RMSNorm output
        l0.q / l0.k / l0.v         [seq, 896|128|128]  projections, bias INCLUDED, pre-RoPE
        l0.q_rope / l0.k_rope      [seq, heads, 64]    after RoPE
        l0.attn_core               [seq, 896]          attention output, before o_proj
        l0.attn_out                [seq, 896]          after o_proj
        l0.post_norm               [seq, 896]          post_attention_layernorm output
        l0.gate / l0.up            [seq, 4864]
        l0.mlp_act                 [seq, 4864]         silu(gate) * up  (input to down_proj)
        l0.mlp_out                 [seq, 896]
        layer.{0..23}              [seq, 896]          residual stream after each decoder layer
        final_norm                 [seq, 896]          after model.norm
        logits_last                [vocab]             logits at the last prompt position
"""

import argparse
import json
import os

import numpy as np
import torch
import transformers
from safetensors.numpy import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

N_GREEDY = 32  # tokens of greedy continuation to record per prompt

LONG_TEXT = (
    "Large language models generate text one token at a time. At each step the model "
    "reads every previous token, computes attention over them, and produces a probability "
    "distribution over its vocabulary. Without caching, the cost of each step grows with the "
    "length of the sequence, because the keys and values for earlier tokens are recomputed "
    "from scratch. A key-value cache stores those tensors after they are first computed, so "
    "each new step only processes the newest token. This trades memory for compute, and on "
    "a CPU the bottleneck usually becomes memory bandwidth: every generated token requires "
    "streaming all of the model's weights from RAM through the cores. Quantization helps "
    "precisely because it shrinks the number of bytes that must be moved per token."
)


def build_prompts(tok):
    chat_msgs = [{"role": "user", "content": "Explain what a KV cache is in one sentence."}]
    # Qwen's template injects a default system prompt; we store the rendered string so the
    # Rust side tokenizes the exact same text instead of re-implementing the template blindly.
    chat_text = tok.apply_chat_template(chat_msgs, tokenize=False, add_generation_prompt=True)
    return {
        "raw_short": "The capital of France is",
        "raw_long": LONG_TEXT,
        "chat": chat_text,
    }


def to_np(t):
    return np.ascontiguousarray(t.detach().float().cpu().numpy())


def first(out):
    # Decoder layers return a tuple in some transformers versions and a tensor in others.
    return out[0] if isinstance(out, (tuple, list)) else out


def register_hooks(model, cap):
    handles = []
    m = model.model
    l0 = m.layers[0]

    def fwd(name):
        def hook(_mod, _inp, out):
            cap[name] = to_np(first(out)[0])  # drop batch dim
        return hook

    def pre(name):
        def hook(_mod, inp):
            cap[name] = to_np(inp[0][0])
        return hook

    handles.append(m.embed_tokens.register_forward_hook(fwd("embed")))
    handles.append(l0.input_layernorm.register_forward_hook(fwd("l0.input_norm")))
    handles.append(l0.self_attn.q_proj.register_forward_hook(fwd("l0.q")))
    handles.append(l0.self_attn.k_proj.register_forward_hook(fwd("l0.k")))
    handles.append(l0.self_attn.v_proj.register_forward_hook(fwd("l0.v")))
    handles.append(l0.self_attn.o_proj.register_forward_pre_hook(pre("l0.attn_core")))
    handles.append(l0.self_attn.o_proj.register_forward_hook(fwd("l0.attn_out")))
    handles.append(l0.post_attention_layernorm.register_forward_hook(fwd("l0.post_norm")))
    handles.append(l0.mlp.gate_proj.register_forward_hook(fwd("l0.gate")))
    handles.append(l0.mlp.up_proj.register_forward_hook(fwd("l0.up")))
    handles.append(l0.mlp.down_proj.register_forward_pre_hook(pre("l0.mlp_act")))
    handles.append(l0.mlp.register_forward_hook(fwd("l0.mlp_out")))
    for i, layer in enumerate(m.layers):
        handles.append(layer.register_forward_hook(fwd(f"layer.{i}")))
    handles.append(m.norm.register_forward_hook(fwd("final_norm")))
    return handles


def rope_reference(model, cap, seq_len):
    """Recompute layer-0 q/k after RoPE using HF's own functions."""
    cfg = model.config
    n_h, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = cfg.hidden_size // n_h
    try:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

        q = torch.from_numpy(cap["l0.q"]).view(1, seq_len, n_h, hd).transpose(1, 2)
        k = torch.from_numpy(cap["l0.k"]).view(1, seq_len, n_kv, hd).transpose(1, 2)
        pos = torch.arange(seq_len).unsqueeze(0)
        cos, sin = model.model.rotary_emb(q, pos)  # [1, seq, hd]
        q_r, k_r = apply_rotary_pos_emb(q, k, cos, sin)
        cap["rope.cos"] = to_np(cos[0])
        cap["rope.sin"] = to_np(sin[0])
        cap["l0.q_rope"] = to_np(q_r[0].transpose(0, 1))  # [seq, heads, hd]
        cap["l0.k_rope"] = to_np(k_r[0].transpose(0, 1))
    except Exception as e:  # internal API moved in some version: don't block the rest
        print(f"  [warn] RoPE reference skipped ({type(e).__name__}: {e})")


@torch.no_grad()
def greedy(model, ids, n):
    """Manual greedy loop: no generation_config (Qwen ships repetition_penalty/top_k/top_p
    defaults that would silently change 'greedy' output), no KV cache, exactly n tokens."""
    out = []
    cur = ids
    for _ in range(n):
        logits = model(input_ids=cur).logits[0, -1]
        nxt = int(torch.argmax(logits))
        out.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen2.5-0.5b-instruct")
    ap.add_argument("--out", default="fixtures")
    ap.add_argument("--revision", default=None, help="HF commit sha the weights came from")
    args = ap.parse_args()

    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    # bf16 -> f32 upcast is exact, so this is the same model at higher compute precision.
    # "dtype" is the current kwarg; older transformers used "torch_dtype".
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float32, attn_implementation="eager"
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32, attn_implementation="eager"
        )
    model.eval()

    p = next(model.parameters())
    assert p.dtype == torch.float32, f"model loaded as {p.dtype}, expected float32"
    tied = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    print(f"tied embeddings: {tied}")  # expect True: there is no lm_head.weight in the file

    manifest = {
        "model": args.model,
        "revision": args.revision,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "dtype": "float32",
        "attn_implementation": "eager",
        "n_greedy": N_GREEDY,
        "prompts": [],
    }

    for name, text in build_prompts(tok).items():
        # add_special_tokens=False: Qwen adds no BOS anyway, but be explicit so Rust matches.
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        seq_len = ids.shape[1]
        print(f"[{name}] {seq_len} tokens")

        cap = {}
        handles = register_hooks(model, cap)
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0]  # [seq, vocab]
        for h in handles:
            h.remove()
        cap["logits_last"] = to_np(logits[-1])
        rope_reference(model, cap, seq_len)

        top = torch.topk(logits[-1], 5)
        print("  top-5 next:", [(tok.decode([int(i)]), round(float(v), 3))
                                for v, i in zip(top.values, top.indices)])

        g = greedy(model, ids, N_GREEDY)
        print("  greedy:", repr(tok.decode(g)))

        fname = f"{name}.safetensors"
        save_file(cap, os.path.join(args.out, fname))
        manifest["prompts"].append({
            "name": name,
            "text": text,
            "input_ids": ids[0].tolist(),
            "greedy_ids": g,
            "fixture": fname,
        })

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"wrote {args.out}/manifest.json")


if __name__ == "__main__":
    main()
