"""Local proxy judge for head-to-head evaluation without the OpenAI API.

The official metric of this project is the win rate against the frozen base model under
GPT-5.4 (see `student_autograder/common.py`). This script reproduces the same protocol with a
local open-weight instruct model so that no API key is needed and no data leaves the machine:

  * the same judge instructions and the same pair prompt as the autograder,
  * every pair is judged twice, with the two responses in both A/B orders,
  * `win_rate_pair_agree`: a row only counts when both orders agree on the winner (the
    autograder's `pair_agree` rule),
  * `win_rate_debiased_all_rows`: every row counts; the log-odds of the two orders are added, which
    cancels the judge's position bias (the main proxy metric, because a local judge has a much
    stronger position bias than GPT-5.4 and the strict rule discards most rows).

Instead of parsing generated JSON, the judge's answer is read from the next-token logits of the
labels "A" and "B" after the forced answer prefix `{"best_label": "`, which is deterministic and
needs one forward pass per order. The autograder's additional `usable_for_training` filter is not
reproduced. Numbers from this script are PROXY win rates; they are not comparable to the GPT-5.4
thresholds of the handout.

Modes:
  policy  student JSONL (row_id, response_text) vs. public_eval base responses -> proxy win rate
  prefs   chosen vs. rejected on public_test_prefs_256.jsonl -> agreement with the GPT-5.4 labels
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from student_autograder.common import JUDGE_INSTRUCTIONS, _build_pair_input

ANSWER_PREFIX = '{"best_label": "'


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class LocalJudge:
    def __init__(self, model_name: str, max_text_chars: int = 6000):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()
        self.max_text_chars = max_text_chars
        ids_a = self.tokenizer.encode("A", add_special_tokens=False)
        ids_b = self.tokenizer.encode("B", add_special_tokens=False)
        if len(ids_a) != 1 or len(ids_b) != 1:
            raise RuntimeError("The labels A and B must be single tokens for this judge model.")
        self.id_a, self.id_b = ids_a[0], ids_b[0]

    @torch.no_grad()
    def prob_first_is_best(self, prompt: str, first: str, second: str) -> float:
        """P(best_label = A | A or B) with `first` shown as [A] and `second` as [B]."""
        user = _build_pair_input(prompt[: self.max_text_chars], "A", first[: self.max_text_chars], "B", second[: self.max_text_chars])
        messages = [{"role": "system", "content": JUDGE_INSTRUCTIONS}, {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + ANSWER_PREFIX
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        logits = self.model(**enc).logits[0, -1].float()
        pair = torch.log_softmax(logits[[self.id_a, self.id_b]], dim=0)
        return float(pair[0].exp().item())

    def judge_pair(self, prompt: str, x: str, y: str) -> Dict[str, Any]:
        """Judge x against y in both orders."""
        p1 = self.prob_first_is_best(prompt, x, y)        # x shown as A
        p2 = 1.0 - self.prob_first_is_best(prompt, y, x)  # x shown as B
        w1, w2 = p1 > 0.5, p2 > 0.5
        return {
            "p_x_pass1": p1,
            "p_x_pass2": p2,
            "p_x_mean": 0.5 * (p1 + p2),
            "pair_agree": w1 == w2,
            "winner": ("x" if w1 else "y") if w1 == w2 else None,
        }


def _logit(p: float) -> float:
    p = min(max(p, 1e-7), 1.0 - 1e-7)
    return math.log(p / (1.0 - p))


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    agree = [r for r in rows if r["pair_agree"]]
    wins = sum(r["winner"] == "x" for r in agree)
    rate = wins / len(agree) if agree else 0.0
    # Debiased decision on every row: add the log-odds of the two orders, so that an additive
    # position bias of the judge cancels; an exact tie (identical responses) counts as half a win.
    z = [_logit(r["p_x_pass1"]) + _logit(r["p_x_pass2"]) for r in rows]
    soft = sum(1.0 if v > 1e-4 else (0.0 if v < -1e-4 else 0.5) for v in z) / n if n else 0.0
    return {
        "count_total_rows": n,
        "count_pair_agree_rows": len(agree),
        "pair_agree_fraction": len(agree) / n if n else 0.0,
        "count_x_wins_pair_agree": wins,
        "win_rate_pair_agree": rate,
        "win_rate_pair_agree_stderr": math.sqrt(rate * (1 - rate) / len(agree)) if agree else 0.0,
        "win_rate_debiased_all_rows": soft,
        "win_rate_debiased_all_rows_stderr": math.sqrt(soft * (1 - soft) / n) if n else 0.0,
        "position_bias_logit": sum(_logit(r["p_x_pass1"]) - _logit(r["p_x_pass2"]) for r in rows) / (2 * n) if n else 0.0,
        "first_position_pick_rate": sum((r["p_x_pass1"] > 0.5) + (r["p_x_pass2"] <= 0.5) for r in rows) / (2 * n) if n else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["policy", "prefs"], required=True)
    ap.add_argument("--judge_model", type=str, default="microsoft/phi-4")
    ap.add_argument("--public_eval_dir", type=str, default="public_eval")
    ap.add_argument("--student_jsonl", type=str, nargs="*", default=[], help="policy mode: one or more submission JSONLs")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    judge = LocalJudge(args.judge_model)
    pub = Path(args.public_eval_dir)

    if args.mode == "prefs":
        prefs = load_jsonl(str(pub / "public_test_prefs_256.jsonl"))
        if args.limit:
            prefs = prefs[: args.limit]
        rows = []
        for r in prefs:
            j = judge.judge_pair(str(r["prompt_text"]), str(r["chosen_text"]).strip(), str(r["rejected_text"]).strip())
            rows.append({"row_id": r["row_id"], **j})
        summary = {"mode": "prefs", "judge_model": args.judge_model, "x": "gpt-5.4 chosen", "y": "gpt-5.4 rejected", **summarize(rows)}
        (out_dir / "prefs_judgments.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        (out_dir / "prefs_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    prompts = load_jsonl(str(pub / "public_test_gen_prompts_128.jsonl"))
    base = {str(r["row_id"]): r for r in load_jsonl(str(pub / "public_test_gen_base_responses_128.jsonl"))}
    if args.limit:
        prompts = prompts[: args.limit]
    for path in args.student_jsonl:
        student = {str(r["row_id"]): r for r in load_jsonl(path)}
        rows = []
        for p in prompts:
            rid = str(p["row_id"])
            s_text = str(student[rid].get("response_text", "")).strip() or "[no response]"
            b_text = str(base[rid].get("response_text", "")).strip()
            j = judge.judge_pair(str(p.get("prompt_text", "")), s_text, b_text)
            rows.append({"row_id": rid, **j})
        name = Path(path).stem
        summary = {"mode": "policy", "judge_model": args.judge_model, "student_jsonl": path, "x": "student", "y": "base", **summarize(rows)}
        (out_dir / f"{name}_judgments.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        (out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(name, json.dumps(summary))


if __name__ == "__main__":
    main()
