"""
Text-to-Motion Inference Script
Usage:
    python infer.py --text "a person walks forward" --output output.pkl

Requires:
    checkpoints/mllm/          - fine-tuned Qwen2.5-VL model
    checkpoints/tokenizer.pt   - FSQ tokenizer checkpoint
    checkpoints/norm_stats.npz - normalization statistics
    checkpoints/smplx_model/   - SMPL-X model files
    src/human_body_prior_repo/support_data/dowloads/V02_05/ - VPoser model
"""

import argparse
import pickle
import sys
import os
import numpy as np
import torch
from collections import OrderedDict
from transformers import AutoTokenizer, AutoConfig, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.tokenizer import FSQAE
from feature_utils import recover_motion
from joint2smplx import fit_smplx_from_joints


# ---------- default paths (override via args) ----------
DEFAULT_MLLM_PATH     = "checkpoints/mllm"
DEFAULT_TOKENIZER_PT  = "checkpoints/tokenizer.pt"
DEFAULT_STATS_NPZ     = "checkpoints/norm_stats.npz"
DEFAULT_SMPLX_PATH    = "checkpoints/smplx_model"

MOTION_TOKEN_CONFIG = {
    "start_id":    136281,
    "end_id":      136282,
    "code_base_id": 136283,
    "vocab_end_id": 151643,
}


# ---------- LLM ----------

def load_mllm(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    return model, tokenizer


def prepare_input(tokenizer, text):
    im_start, im_end = "<|im_start|>", "<|im_end|>"
    prefix = tokenizer.encode(f"{im_start}user\nGenerate a motion code sequence for the following action: ", add_special_tokens=False)
    desc   = tokenizer.encode(text)
    suffix = tokenizer.encode(f"{im_end}\n{im_start}assistant\n", add_special_tokens=False)
    return torch.tensor([prefix + desc + suffix])


def make_position_ids(input_ids):
    B, L = input_ids.shape
    pos = torch.zeros_like(input_ids, dtype=torch.long)
    cfg = MOTION_TOKEN_CONFIG
    for i in range(B):
        text_pos, motion_pos, in_motion = 0, 0, False
        for j in range(L):
            tid = input_ids[i, j].item()
            if in_motion:
                pos[i, j] = motion_pos
                motion_pos += 1
                if tid == cfg["end_id"]:
                    in_motion = False
            else:
                pos[i, j] = text_pos
                text_pos += 1
                if tid == cfg["start_id"]:
                    in_motion = True
                    motion_pos = 0
    return pos.view(1, B, L).expand(3, -1, -1)


def generate_motion_tokens(mllm, tokenizer, text, device, max_new_tokens=512):
    cfg = MOTION_TOKEN_CONFIG
    input_ids = prepare_input(tokenizer, text).to(device)
    start_tok = torch.tensor([[cfg["start_id"]]], device=device)
    input_ids = torch.cat([input_ids, start_tok], dim=1)

    prompt_len = input_ids.shape[1]
    position_ids = make_position_ids(input_ids).to(device)

    with torch.no_grad():
        out = mllm(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=True,
            cache_position=torch.arange(prompt_len, device=device),
        )
        past_kv = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    generated = []
    for step in range(max_new_tokens):
        tid = next_tok.item()
        if tid in (tokenizer.eos_token_id, cfg["end_id"]):
            break
        generated.append(tid)
        with torch.no_grad():
            out = mllm(
                input_ids=next_tok,
                position_ids=torch.tensor([[[step]]], device=device).expand(3, 1, 1),
                use_cache=True,
                past_key_values=past_kv,
                cache_position=torch.tensor([prompt_len + step], device=device),
            )
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    motion_codes = [
        tid - cfg["code_base_id"]
        for tid in generated
        if cfg["code_base_id"] <= tid <= cfg["vocab_end_id"]
    ]
    return motion_codes


# ---------- FSQ decoder ----------

def load_norm_stats(stats_path, device):
    data = np.load(stats_path)
    min_vals = torch.from_numpy(data['min_vals']).float().to(device)
    value_range = torch.from_numpy(data['value_range']).float().to(device)
    return min_vals, value_range


def denormalize(data: torch.Tensor, min_vals, value_range) -> torch.Tensor:
    return (data + 1) * value_range / 2 + min_vals

def load_fsq(ckpt_path, device):
    model = FSQAE().to(device)
    sd = torch.load(ckpt_path, map_location=device)["model_state_dict"]
    new_sd = OrderedDict()
    for k, v in sd.items():
        new_sd[k[7:] if k.startswith("module.") else k] = v
    model.load_state_dict(new_sd)
    model.eval()
    return model


def decode_tokens(fsq_model, min_vals, value_range, tokens, smplx_path, device):
    tokens_t = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        z_q = fsq_model.quantize.indices_to_codes(tokens_t)
        recon = fsq_model.decode(z_q)
    rec_data = denormalize(recon, min_vals, value_range).cpu().numpy().reshape(-1, 85)
    joints = recover_motion(rec_data)
    pointcloud = torch.from_numpy(joints).float().to(device).view(-1, 28, 3)
    return fit_smplx_from_joints(pointcloud, num_iters=100, model_path=smplx_path, device=str(device))


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",          type=str, required=True)
    parser.add_argument("--output",        type=str, default="output.pkl")
    parser.add_argument("--mllm_path",     type=str, default=DEFAULT_MLLM_PATH)
    parser.add_argument("--tokenizer_pt",  type=str, default=DEFAULT_TOKENIZER_PT)
    parser.add_argument("--stats_npz",     type=str, default=DEFAULT_STATS_NPZ)
    parser.add_argument("--smplx_path",    type=str, default=DEFAULT_SMPLX_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[1/4] Loading MLLM from {args.mllm_path} ...")
    mllm, tokenizer = load_mllm(args.mllm_path, device)

    print(f"[2/4] Loading FSQ tokenizer from {args.tokenizer_pt} ...")
    fsq_model = load_fsq(args.tokenizer_pt, device)
    min_vals, value_range = load_norm_stats(args.stats_npz, device)

    print(f"[3/4] Generating motion tokens for: '{args.text}'")
    tokens = generate_motion_tokens(mllm, tokenizer, args.text, device)
    if not tokens:
        print("ERROR: MLLM produced no valid motion tokens.")
        return
    print(f"      Generated {len(tokens)} tokens.")

    print("[4/4] Decoding tokens to SMPL-X ...")
    smplx_params = decode_tokens(fsq_model, min_vals, value_range, tokens, args.smplx_path, device)

    with open(args.output, "wb") as f:
        pickle.dump(smplx_params, f)
    print(f"Done. Saved to {args.output}")


if __name__ == "__main__":
    main()
