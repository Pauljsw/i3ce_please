#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scaffold LoRA Validation Script for ShapeLLM

- Loads base model + LoRA adapter
- Runs inference on validation JSON created with your synthetic generator
- Auto-detects task type from prompt pattern
- Parses ground-truth from assistant text
- Computes metrics:
  * Missing: yes/no acc, type acc
  * Referring / Expected box: 3D IoU, corner L2-RMSE
  * Listing: set Precision/Recall/F1

Author: your friendly AI
"""

import os
import re
import json
import math
import argparse
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

import torch
from transformers import TextStreamer
from peft import PeftModel
# Use the same builder/utils as cli.py for compatibility
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import POINT_TOKEN_INDEX, DEFAULT_POINT_TOKEN, DEFAULT_PT_START_TOKEN, DEFAULT_PT_END_TOKEN
from llava.mm_utils import load_pts, process_pts, tokenizer_point_token, get_model_name_from_path, KeywordsStoppingCriteria


# -----------------------------
# Utility: task type detection
# -----------------------------
def detect_task(prompt: str) -> str:
    p = prompt.lower()
    if "any missing member" in p:
        # missing check, may require expected box if mentioned
        return "missing_check"
    if "referring:" in p and "return the box" in p:
        return "referring_box"
    if "list all missing" in p:
        return "list_missing"
    if "why is" in p or "dangerous" in p:
        return "risk_reasoning"
    # fallback
    return "unknown"


# -----------------------------
# Parsing helpers
# -----------------------------
_fnum = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
_corner_pat = re.compile(r"\[\s*(" + _fnum + r")\s*,\s*(" + _fnum + r")\s*,\s*(" + _fnum + r")\s*\]")

def extract_corners(text: str) -> Optional[np.ndarray]:
    """
    Extract up to 8 corners from text like [[x,y,z], [x,y,z], ...].
    Returns (8,3) float array if found, else None.
    """
    triples = _corner_pat.findall(text)
    if len(triples) >= 8:
        pts = np.array(triples[:8], dtype=float)
        return pts
    return None

_missing_yes_pat = re.compile(r"\byes\b", re.IGNORECASE)
_missing_no_pat  = re.compile(r"\bno\b", re.IGNORECASE)
_type_pat = re.compile(r"(vertical|ledger|transom|brace|deck|ladder)", re.IGNORECASE)
_face_pat = re.compile(r"\b(inner|outer)\b", re.IGNORECASE)
_bay_pat  = re.compile(r"\bbay\s+(\d+)\b", re.IGNORECASE)
_level_pat= re.compile(r"\blevel\s+(\d+)\b", re.IGNORECASE)

def parse_missing_answer(text: str) -> Dict[str, Any]:
    """
    Parse a typical missing-check answer like:
    'Yes. Missing transom at Bay 0 Level 1 (outer). Expected bbox corners: [[...]]'
    """
    d = {"is_missing": None, "type": None, "bay": None, "level": None, "face": None, "box": None}
    if _missing_yes_pat.search(text):
        d["is_missing"] = True
    elif _missing_no_pat.search(text):
        d["is_missing"] = False

    mtype = _type_pat.search(text)
    if mtype:
        d["type"] = mtype.group(1).lower()

    bay = _bay_pat.search(text)
    if bay:
        d["bay"] = int(bay.group(1))

    lvl = _level_pat.search(text)
    if lvl:
        d["level"] = int(lvl.group(1))

    face = _face_pat.search(text)
    if face:
        d["face"] = face.group(1).lower()

    box = extract_corners(text)
    if box is not None:
        d["box"] = box
    return d


def try_parse_json_list(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    For 'List all missing members...' answers. Try to locate a JSON list.
    """
    # heuristic: find first '[' ... ']'
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end+1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, list):
                return obj
        except Exception:
            return None
    return None


# -----------------------------
# Box metrics
# -----------------------------
def aabb_from_corners(c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given (N,3) corners, return (mins, maxs)"""
    mins = c.min(axis=0)
    maxs = c.max(axis=0)
    return mins, maxs

def iou3d_from_corners(c1: np.ndarray, c2: np.ndarray) -> float:
    """Axis-aligned IoU in 3D from corner sets."""
    m1, M1 = aabb_from_corners(c1)
    m2, M2 = aabb_from_corners(c2)
    inter_min = np.maximum(m1, m2)
    inter_max = np.minimum(M1, M2)
    inter = np.maximum(0.0, inter_max - inter_min)
    inter_vol = inter[0]*inter[1]*inter[2]
    vol1 = np.prod(M1 - m1)
    vol2 = np.prod(M2 - m2)
    union = vol1 + vol2 - inter_vol
    if union <= 0:
        return 0.0
    return float(inter_vol / union)

def corner_rmse(c1: np.ndarray, c2: np.ndarray) -> float:
    """
    Corner-wise RMSE with greedy matching (robust to ordering differences).
    (Hungarian would be cleaner; greedy is OK for evaluation signal.)
    """
    c1 = c1.copy()
    c2 = c2.copy()
    used = np.zeros(len(c2), dtype=bool)
    dists = []
    for p in c1:
        idx = np.argmin(((c2 - p)**2).sum(axis=1) + 1e9*used.astype(float))
        used[idx] = True
        dists.append(np.sqrt(((c2[idx] - p)**2).sum()))
    return float(np.sqrt(np.mean(np.square(dists))))


# -----------------------------
# Inference (matches cli.py)
# -----------------------------
def run_one(model, tokenizer, pts: np.ndarray, prompt_text: str,
            with_pt_tokens: bool, temperature=0.2, max_new_tokens=512) -> str:
    from llava.constants import DEFAULT_POINT_TOKEN, DEFAULT_PT_START_TOKEN, DEFAULT_PT_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from transformers import TextStreamer
    import torch

    # (1) 프롬프트에 이미 <point> 혹은 <pt_start><point><pt_end> 가 있으면 추가 삽입 금지
    if (DEFAULT_POINT_TOKEN in prompt_text) or (DEFAULT_PT_START_TOKEN in prompt_text):
        with_pt_tokens = False

    conv_mode = "vicuna_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    # (2) 첫 턴 입력 구성 (중복 없이 정확히 한 번만 point 토큰 포함)
    if with_pt_tokens:
        if getattr(model.config, "mm_use_pt_start_end", False):
            q = DEFAULT_PT_START_TOKEN + DEFAULT_POINT_TOKEN + DEFAULT_PT_END_TOKEN + "\n" + prompt_text
        else:
            q = DEFAULT_POINT_TOKEN + "\n" + prompt_text
    else:
        q = prompt_text

    conv.append_message(roles[0], q)
    conv.append_message(roles[1], None)
    prompt = conv.get_prompt()

    # (3) 토크나이즈
    input_ids = tokenizer_point_token(prompt, tokenizer, POINT_TOKEN_INDEX,
                                      return_tensors='pt').unsqueeze(0).to(model.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # (4) 포인트 전처리 (모델과 같은 디바이스/dtype으로)
    pts_tensor = None
    if pts is not None:
        pts_tensor = process_pts(pts, model.config).unsqueeze(0).to(model.device, dtype=torch.float16)

    # (5) 생성
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            points=pts_tensor,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

    out = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    return out



# -----------------------------
# Main
# -----------------------------
def main(args):
    disable_torch_init()

    # Load model (base + LoRA)
    model_name = os.path.basename(args.lora_path.rstrip("/"))
    tokenizer, model, _ = load_pretrained_model(
        args.lora_path,  # model_path points to LoRA checkpoint dir
        args.model_base, # model_base is the original base model
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device
    )

    # Vision tower init (same as training/cli)
    # Note: load_pretrained_model already wires vision tower based on config; ensure files exist
    assert os.path.exists(args.vision_cfg),  f"vision cfg not found: {args.vision_cfg}"
    assert os.path.exists(args.vision_ckpt), f"vision ckpt not found: {args.vision_ckpt}"

    # Load val JSON
    with open(args.val_json, "r") as f:
        val_data = [json.loads(line) if line.strip().startswith("{") and not args.json_array else None for line in f] if not args.json_array else json.load(f)
    if not args.json_array:
        val_data = [x for x in val_data if x is not None]

    # Metrics accumulators
    n_missing = n_missing_correct_yn = n_missing_correct_type = 0
    n_box = iou_sum = rmse_sum = 0
    n_list = f1_num = prec_sum = rec_sum = 0

    preds_dump = []

    # Iterate
    total = len(val_data) if args.max_samples <= 0 else min(args.max_samples, len(val_data))
    for idx in range(total):
        sample = val_data[idx]
        sid = sample.get("id", f"idx_{idx}")
        pcs_file = sample.get("point") or sample.get("pts") or sample.get("point_path")
        assert pcs_file, f"No 'point' file in sample {sid}"

        pts_path = pcs_file
        if not os.path.isabs(pts_path):
            pts_path = os.path.join(args.pcs_dir, pcs_file)

        # Load points (expects .npy with N,3 or N,6)
        pts = load_pts(pts_path)  # same util as cli.py; returns Nx(>=3)

        # Each sample expected: conversations: [{"from":"human","value":...},{"from":"gpt","value":...}]
        convs = sample.get("conversations", [])
        # Use the first pair only (your dataset uses one QA per line)
        gt_question = None
        gt_answer = None
        for turn in convs:
            if turn.get("from") == "human" and gt_question is None:
                gt_question = turn.get("value", "")
            elif turn.get("from") in ("gpt", "assistant") and gt_answer is None:
                gt_answer = turn.get("value", "")
        if gt_question is None or gt_answer is None:
            continue

        task = detect_task(gt_question)

        # Run model
        pred_text = run_one(model, tokenizer, pts, gt_question, with_pt_tokens=True,
                            temperature=args.temperature, max_new_tokens=args.max_new_tokens)

        # Parse ground-truth and prediction
        res = {"id": sid, "task": task, "question": gt_question, "gt": gt_answer, "pred": pred_text}

        if task == "missing_check":
            n_missing += 1
            gt = parse_missing_answer(gt_answer)
            pr = parse_missing_answer(pred_text)

            # yes/no
            if gt["is_missing"] is not None and pr["is_missing"] is not None and gt["is_missing"] == pr["is_missing"]:
                n_missing_correct_yn += 1
            # type (only if missing)
            if gt["is_missing"]:
                if (gt["type"] is not None) and (pr["type"] is not None) and (gt["type"] == pr["type"]):
                    n_missing_correct_type += 1

            # if both have boxes, compute IoU/RMSE
            if gt.get("box") is not None and pr.get("box") is not None:
                iou = iou3d_from_corners(gt["box"], pr["box"])
                rmse = corner_rmse(gt["box"], pr["box"])
                iou_sum += iou
                rmse_sum += rmse
                n_box += 1

            res.update({"gt_parsed": gt, "pred_parsed": pr})

        elif task == "referring_box":
            gt_box = extract_corners(gt_answer)
            pr_box = extract_corners(pred_text)
            if gt_box is not None and pr_box is not None:
                iou = iou3d_from_corners(gt_box, pr_box)
                rmse = corner_rmse(gt_box, pr_box)
                iou_sum += iou
                rmse_sum += rmse
                n_box += 1
            res.update({"gt_box": gt_box.tolist() if gt_box is not None else None,
                        "pr_box": pr_box.tolist() if pr_box is not None else None})

        elif task == "list_missing":
            gt_list = try_parse_json_list(gt_answer) or []
            pr_list = try_parse_json_list(pred_text) or []

            # Reduce to sets of (type,bay,level,face) ignoring box for F1
            def reduce_set(lst):
                S = set()
                for it in lst:
                    try:
                        S.add((str(it["type"]).lower(), int(it["bay"]), int(it["level"]), str(it["face"]).lower()))
                    except Exception:
                        continue
                return S

            Sg = reduce_set(gt_list)
            Sp = reduce_set(pr_list)
            tp = len(Sg & Sp)
            fp = len(Sp - Sg)
            fn = len(Sg - Sp)
            prec = tp / (tp + fp) if tp + fp > 0 else 0.0
            rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1   = 2*prec*rec / (prec+rec) if prec+rec > 0 else 0.0
            prec_sum += prec
            rec_sum  += rec
            f1_num   += f1
            n_list   += 1

            res.update({"gt_list": gt_list, "pr_list": pr_list, "prec": prec, "rec": rec, "f1": f1})

        else:
            # risk_reasoning or unknown → 스코어 생략(텍스트 유사도 지표를 추가하고 싶으면 여기에서 계산)
            pass

        preds_dump.append(res)

    # Aggregate metrics
    out = {}
    if n_missing > 0:
        out["missing_yesno_acc"]  = round(n_missing_correct_yn / n_missing, 4)
        out["missing_type_acc"]   = round(n_missing_correct_type / max(1, n_missing), 4)
    if n_box > 0:
        out["box_mean_iou3d"]     = round(iou_sum / n_box, 4)
        out["box_mean_corner_rmse"]= round(rmse_sum / n_box, 4)
    if n_list > 0:
        out["list_mean_precision"] = round(prec_sum / n_list, 4)
        out["list_mean_recall"]    = round(rec_sum  / n_list, 4)
        out["list_mean_f1"]        = round(f1_num  / n_list, 4)

    print("\n=== Validation Summary ===")
    for k, v in out.items():
        print(f"{k:>26s}: {v}")
    print(f"samples_evaluated: {total} | box_pairs: {n_box} | list_cases: {n_list} | missing_cases: {n_missing}")

    # Save raw preds if requested
    if args.save_preds:
        os.makedirs(os.path.dirname(args.save_preds), exist_ok=True)
        with open(args.save_preds, "w") as f:
            json.dump({"metrics": out, "preds": preds_dump}, f, indent=2)
        print(f"\nSaved predictions to: {args.save_preds}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-base", type=str, required=True, help="qizekun/ShapeLLM_7B_gapartnet_v1.0")
    ap.add_argument("--lora-path", type=str, required=True, help="LoRA adapter dir (contains adapter_model.bin)")
    ap.add_argument("--val-json", type=str, required=True)
    ap.add_argument("--pcs-dir", type=str, required=True)
    ap.add_argument("--vision-cfg", type=str, required=True)
    ap.add_argument("--vision-ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=-1, help="<=0 means all")
    ap.add_argument("--save-preds", type=str, default="./val_preds_scaffold.json")
    ap.add_argument("--load-8bit", action="store_true")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--json-array", action="store_true", help="Set if your val json is a single JSON array not jsonl")
    args = ap.parse_args()
    main(args)
