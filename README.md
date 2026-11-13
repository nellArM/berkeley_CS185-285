# Berkeley CS 185/285 Deep Reinforcement Learning (Spring 2026) — 作業與期末專題 / Assignments and Final Projects

本資料夾收錄 CS 185/285 五次作業與兩個期末專題的解答與報告。所有內容為**自學用途**，並非正式修課繳交；所有實驗皆在本機 GPU（NVIDIA RTX A6000）執行，wandb 為離線模式。每個子資料夾都有英文報告 `report.pdf`。

This folder contains solutions and reports for the five assignments and both final projects of CS 185/285. Everything is for **self-study**, not an official course submission; all experiments ran on local GPUs (NVIDIA RTX A6000) with wandb in offline mode. Every sub-folder has an English report, `report.pdf`.

## 總覽 / Overview

| 資料夾 Folder | 主題 Topic | 主要結果 Main result |
|---|---|---|
| `hw1/` | 模仿學習 Imitation learning (Push-T) | MSE 0.60, flow matching 0.85 |
| `hw2/` | 策略梯度 Policy gradients | LunarLander 226.5 (GAE λ = 0.98) |
| `hw3/` | Q-learning 與 SAC / Q-learning and SAC | all targets met (Hopper: 1 of 3 seeds) |
| `hw4/` | LLM 的強化學習 RL for LLMs (GR-REINFORCE, GRPO) | math_hard GRPO eval 0.40 |
| `hw5/` | 離線強化學習 Offline RL (SAC+BC, IQL, FQL) | all six targets met |
| `final_project_offline_online/` | 離線到線上 RL / Offline-to-online RL | cube-double 0.96, antsoccer 0.88 |
| `final_project_llm_rl/` | LLM 的 RLHF / RLHF for instruction following | proxy-judge win rates, see below |

---

## hw1 — 模仿學習 / Imitation Learning

**目的。** 在 Push-T 任務上用行為複製學習策略，比較兩種「動作區塊」(action chunking) 策略：以 MSE 迴歸直接輸出動作，以及用 flow matching 產生動作，理解多峰動作分佈下生成式策略的優勢。

**Purpose.** Learn a policy for the Push-T task by behaviour cloning and compare two action-chunking policies: direct MSE regression and a flow-matching (generative) policy, to see why a generative policy helps when the demonstrated actions are multi-modal.

**結果 / Results.** MSE policy: final eval reward 0.595 (target ≥ 0.5). Flow matching: 0.845 (target ≥ 0.7).

**內容 / Contents.** `src/` (code), `report.pdf`.

## hw2 — 策略梯度 / Policy Gradients

**目的。** 實作 REINFORCE 系列的策略梯度：reward-to-go、優勢標準化、神經網路 baseline 與 GAE，並在 CartPole、HalfCheetah、LunarLander、InvertedPendulum 上觀察各技巧對變異數與學習穩定度的影響。

**Purpose.** Implement policy gradients with reward-to-go, advantage normalisation, a neural-network baseline and GAE, and study how each variance-reduction technique affects learning on CartPole, HalfCheetah, LunarLander and InvertedPendulum.

**結果 / Results.** All eight CartPole configurations reach the maximum return of 200; advantage normalisation is the most influential factor. LunarLander with GAE peaks at 226.5 with λ = 0.98 (threshold 150). Details for HalfCheetah and InvertedPendulum are in the report.

**內容 / Contents.** `src/` (code), `report.pdf`.

## hw3 — Q-learning 與 Actor-Critic / Q-Learning and Actor-Critic

**目的。** 實作 DQN 與 double DQN（離散動作），以及 SAC（連續動作）：固定與自動調整溫度、clipped double-Q，並理解 Q 值高估問題。

**Purpose.** Implement DQN and double DQN for discrete actions, and SAC for continuous control (fixed and auto-tuned temperature, clipped double-Q), and study over-estimation bias.

**結果 / Results.**

| Experiment | Target | Result |
|---|---|---|
| DQN, CartPole-v1 | 500 | 500 |
| Double DQN, LunarLander-v2 | 200 | max 256 |
| Double DQN, MsPacman | about 1500 | max 2232 |
| SAC, HalfCheetah-v4 | ≥ 6000 | max 10,355 |
| SAC clipped double-Q, Hopper-v4 | 1500 | max 3474 with seed 3; seeds 1 and 2 reach 1034 and 1464 |

Hopper 的結果對亂數種子很敏感，三個種子中只有一個達標，報告中如實列出全部三個種子。 The Hopper result depends strongly on the seed; only one of three seeds reaches the target, and the report lists all three.

**內容 / Contents.** `src/` (code), `report.pdf`.

## hw4 — LLM 的強化學習 / RL for LLMs

**目的。** 以 Qwen2.5-1.5B-Instruct + LoRA 為策略，實作 GR-REINFORCE 與 GRPO（群組相對優勢、PPO 式裁剪、取樣式 KL 懲罰），在格式複製任務與數學題任務上訓練。

**Purpose.** With Qwen2.5-1.5B-Instruct + LoRA as the policy, implement GR-REINFORCE and GRPO (group-relative advantages, PPO-style clipping, sampled KL penalty) and train on a format-copy task and a hard-math task.

**結果 / Results.** format_copy: eval exact match 1.00 with both algorithms. math_hard: GRPO 0.400 (expectation > 0.37, met); GR-REINFORCE 0.287 (handout range 0.29–0.31, 0.003 below; reported as "close", not met). Runs were local instead of Modal, at full length.

**內容 / Contents.** `hw4/` (code), `scripts/` (local launcher, plotting and Modal entry point), `report.pdf`.

## hw5 — 離線強化學習 / Offline RL

**目的。** 只用固定資料集學習策略：實作 SAC+BC、IQL 與 FQL (flow Q-learning)，在 OGBench 的 cube-single 與 antsoccer-arena 上調整行為正則化強度 α，理解分佈外動作造成的價值高估以及各方法的處理方式。

**Purpose.** Learn from a fixed dataset only: implement SAC+BC, IQL and FQL (flow Q-learning), tune the behaviour-regularisation strength α on OGBench cube-single and antsoccer-arena, and understand how each method avoids over-estimated values of out-of-distribution actions.

**結果 / Results (final success rate at 1M steps).** SAC+BC 1.00 / 0.16, IQL 0.80 / 0.28, FQL 1.00 / 0.28 (cube-single / antsoccer-arena); all six handout targets are met.

**內容 / Contents.** `src/` (code), `report.pdf`.

---

## final_project_offline_online — 離線到線上強化學習 / Offline-to-Online RL

**目的。** 離線預訓練的策略在開始線上互動後常會先變差甚至崩潰。本專題先重現這個現象（SAC+BC、FQL 的 naive 微調），再實作五種改良方法（保留離線資料、WSRL、IFQL + rejection sampling、DSRL、QSM），最後設計自己的方法並與全部基線比較。

**Purpose.** A policy that was pre-trained offline often gets worse, or collapses, when online interaction starts. The project reproduces this with naive fine-tuning of SAC+BC and FQL, implements five remedies (retained offline data, WSRL, IFQL with rejection sampling, DSRL, QSM), and then designs a method and compares it with all baselines.

**方法 / Method (FQL-X).** FQL 的損失不變；線上階段把全部 100 萬筆離線資料留在 replay buffer、均勻取樣，並在每個環境步做 4 次梯度更新。 FQL's losses are unchanged; in the online phase the whole offline dataset (1M transitions) stays in the replay buffer with uniform sampling, and four gradient updates are made per environment step.

**結果 / Results (500K offline + 100K online steps, mean of two seeds, 25 evaluation episodes per seed).**

| Method | cube-double | antsoccer-arena |
|---|---|---|
| FQL (naive) | 0.14 | 0.00 |
| FQL + offline data | 0.88 | 0.80 |
| FQL + WSRL | 0.66 | 0.52 |
| IFQL | 0.08 | 0.02 |
| DSRL | 0.12 | 0.02 |
| QSM | 0.00 | 0.00 |
| **FQL-X (ours)** | **0.96** (target > 0.95) | **0.88** (target > 0.45) |

注意事項 / Caveats: cube-double 僅比門檻多一個成功回合（50 回合中 48 回合），與最強基線的差距在評估雜訊範圍內。QSM 在 antsoccer-arena 依講義設定（兩個 critic 取最小值）未達離線門檻 0.35，改用平均值作為 target 後才達標（0.36），報告中有完整說明。 The cube-double result is one episode above the threshold (48 of 50), and the gap to the strongest baseline is within evaluation noise. QSM on antsoccer-arena misses the offline threshold of 0.35 with the handout's minimum target and reaches it (0.36) only with a mean target; the report documents this.

**內容 / Contents.** `s1/`, `s2_1/`, `s2_2/`, `s2_3/`, `s3/` (code of each section, each a separate `uv` project), `report.pdf`.

## final_project_llm_rl — 開放式指令遵循的 RLHF / RLHF for Open-Ended Instruction Following

**目的。** 用人類偏好資料改進語言模型的回答品質。Part 1 實作離線偏好最佳化（DPO、IPO、AOT）、Bradley–Terry reward model，以及以 reward model 為獎勵的線上方法（GRPO、DrGRPO、GSPO）；Part 2 為自訂的改良研究。評量方式是與未微調的基礎模型逐題比較回答的勝率。

**Purpose.** Improve the answers of a language model with preference data. Part 1 implements offline preference optimisation (DPO, IPO, AOT), a Bradley–Terry reward model, and reward-model-based online methods (GRPO, DrGRPO, GSPO); Part 2 is an open-ended investigation. The metric is the head-to-head win rate against the base model.

**重要說明 / Important.** 官方評量使用 GPT-5.4 (OpenAI API) 當裁判；本解答沒有使用 API，改用本機開源模型 `microsoft/phi-4` 作為**代理裁判**。代理裁判與資料集中 GPT-5.4 標註的一致率為 0.82，有明顯的位置偏差（已用雙向評比抵銷）以及偏好較長回答的傾向。因此下列勝率**不能**與講義的門檻直接比較。 The official metric uses GPT-5.4 as the judge; this solution uses a local open-weight model (`microsoft/phi-4`) as a **proxy judge** instead. It agrees with the GPT-5.4 labels of the benchmark on 82% of pairs, has a strong position bias (cancelled by judging both orders) and prefers longer responses. The win rates below are therefore **not** comparable with the handout thresholds.

**結果 / Results (proxy win rate against the base model, 128 prompts, standard error about 0.04).**

| Part 1 | Win rate | Part 2 | Win rate |
|---|---|---|---|
| DPO | 0.67 | DPO + confidence weighting (submitted `offline_best`) | 0.73 |
| IPO | 0.65 | DPO + DPO-Positive penalty / label smoothing | 0.61–0.65 |
| AOT | 0.80 | Reward-model reranking of DPO/IPO/AOT | 0.71 |
| GRPO | 0.58 | GRPO, 75 iterations | 0.68 |
| DrGRPO | 0.38 | GRPO + rank-only advantages (submitted `online_best`) | 0.68 |
| GSPO | 0.57 | GRPO + pessimistic reward-model ensemble | 0.67 |

Reward model 在公開測試配對上的準確率為 0.832（門檻 0.74，此項不需裁判）。 The reward model reaches a pair accuracy of 0.832 on the public pairs (threshold 0.74; no judge needed).

主要發現 / Main findings: reward model 的分數無法預測裁判勝率（DrGRPO 的 reward model 分數最高，勝率卻低於基礎模型）；Part 2 的離線改良都沒有超過 Part 1 的 AOT；線上方法的進步主要來自較長的訓練，rank-only advantage 則讓訓練穩定（KL 不再暴衝）。 The reward-model score does not predict the judged win rate (DrGRPO has the highest score and loses to the base model); no Part 2 offline idea beats Part 1 AOT; the online gain comes mainly from longer training, and rank-only advantages make training stable (no KL spikes).

**個資處理 / Personal data.** 課程提供的公開評測提示中有一則（來自 WildChat 公開資料集）含有第三方的履歷個資（姓名、電話、電子郵件、地址），因此本資料夾不包含課程提供的 `public_eval/` 與 `dataset/` 資料（重跑時請從課程原始碼庫取得）。 One prompt of the course's public evaluation set (from the public WildChat dataset) contains a third party's résumé details (name, phone, e-mail, address), so this folder does not include the course-provided `public_eval/` and `dataset/` data (obtain them from the course repository to re-run).

**內容 / Contents.** `code/` (training code, scripts and the local autograder), `report.pdf`.

---

## 重現方式 / Reproduction

每份報告的最後都列出實際執行的指令。環境皆以 [`uv`](https://docs.astral.sh/uv/) 管理：在對應的作業目錄執行 `uv sync` 後即可用報告中的指令重跑。常用環境變數： Each report ends with the exact commands that were run. Environments are managed with `uv`: run `uv sync` in the assignment directory and then the commands of the report. Common environment variables:

```bash
export WANDB_MODE=offline   # no wandb account needed
export MUJOCO_GL=egl        # headless MuJoCo rendering (hw3, hw5, offline-to-online project)
```

報告 PDF 是以下列指令由 Markdown 轉出；Markdown 原始檔與圖片未收錄於本資料夾，因此無法直接重新產生 PDF。 The PDFs were built from Markdown with the command below; the Markdown sources and figures are not included in this folder, so the PDFs cannot be rebuilt from it.

```bash
uv run --no-project --python 3.11 --with pypandoc_binary --with typst tools/md2pdf.py <report.md> <report.pdf>
```

## 隱私處理 / Privacy

公開前已檢查本資料夾內所有檔案（含二進位檔）。處理項目： All files in this folder (including binary files) were checked before release. Changes made:

- `final_project_llm_rl/`：不收錄含第三方個資的課程評測資料（見該節說明）。 The course evaluation data that contains third-party personal data is not included (see that section).

## 授權 / License

本資料夾中我撰寫的程式與報告以 MIT 授權釋出（見 `LICENSE`）。課程提供的起始程式碼、講義與資料集版權屬原作者所有。 My own code and reports in this folder are released under the MIT License (see `LICENSE`). The starter code, handouts and datasets provided by the course remain the property of their authors.
