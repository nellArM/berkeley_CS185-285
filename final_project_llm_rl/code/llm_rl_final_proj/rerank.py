"""Part 2 (offline): reward-model reranking over a fixed family of candidate policies.

Every candidate JSONL holds one greedy response per public prompt (built with
`build_policy_submission.py` under the fixed benchmark decoding settings). For each prompt the
response with the highest reward-model score is written to the output JSONL. The script also saves
the per-candidate scores, which are used in the report to compare reward-model scores with the
judge's win rates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from llm_rl_final_proj.models.load import load_reward_model_and_tokenizer, resolve_adapter_path
from llm_rl_final_proj.reward_model.evaluation import score_prompt_response_pairs


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter_path", type=str, required=True, help="reward-model adapter")
    ap.add_argument("--prompts_jsonl", type=str, default="public_eval/public_test_gen_prompts_128.jsonl")
    ap.add_argument("--candidates", type=str, nargs="+", required=True, help="candidate JSONLs (all are scored)")
    ap.add_argument("--select_from", type=str, nargs="*", default=None, help="subset of candidate stems used for selection (default: all)")
    ap.add_argument("--output_jsonl", type=str, default="")
    ap.add_argument("--scores_json", type=str, required=True)
    ap.add_argument("--max_prompt_tokens", type=int, default=700)
    ap.add_argument("--max_response_tokens", type=int, default=512)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    loaded = load_reward_model_and_tokenizer(
        args.model_name, device=device, dtype=dtype, adapter_path=resolve_adapter_path(args.adapter_path)
    )
    prompts = _load_jsonl(args.prompts_jsonl)
    scores: Dict[str, List[float]] = {}
    responses: Dict[str, List[Dict[str, Any]]] = {}
    for path in args.candidates:
        name = Path(path).stem
        by_id = {str(r["row_id"]): r for r in _load_jsonl(path)}
        rows = [
            {
                "row_id": str(p["row_id"]),
                "prompt_messages": p["prompt_messages"],
                "prompt_text": p.get("prompt_text", ""),
                "response_text": str(by_id[str(p["row_id"])].get("response_text", "")).strip(),
            }
            for p in prompts
        ]
        scores[name] = [
            float(x)
            for x in score_prompt_response_pairs(
                loaded.model,
                loaded.tokenizer,
                rows,
                max_prompt_tokens=args.max_prompt_tokens,
                max_response_tokens=args.max_response_tokens,
                per_device_batch_size=8,
                device=device,
            )
        ]
        responses[name] = [by_id[str(p["row_id"])] for p in prompts]
        print(f"{name}: mean reward-model score {sum(scores[name]) / len(scores[name]):.3f}")

    out: Dict[str, Any] = {"row_ids": [str(p["row_id"]) for p in prompts], "scores": scores}
    if args.output_jsonl:
        family = args.select_from or list(scores)
        picks: List[str] = []
        lines = []
        for i, p in enumerate(prompts):
            best = max(family, key=lambda n: scores[n][i])
            picks.append(best)
            row = dict(responses[best][i])
            row["row_id"] = str(p["row_id"])
            lines.append(json.dumps(row, ensure_ascii=False))
        Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_jsonl).write_text("\n".join(lines) + "\n", encoding="utf-8")
        out["select_from"] = family
        out["picks"] = picks
        print("selection counts:", {n: picks.count(n) for n in family})
    Path(args.scores_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.scores_json).write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    main()
