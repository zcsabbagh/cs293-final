# CS293 Final: Fine-Tuning LLMs for CCSS Standard Prediction

## Task

Given a K-12 math problem, predict which Common Core State Standards (CCSS) it addresses ("Addressing") and which prerequisite standards it builds on ("Building On").

## Dataset

**MathFish** (allenai/mathfish) from HuggingFace.

- 13,065 train / 4,355 test problems total
- ~48.7% have empty standards — filtered out
- **Addressing task**: 4,960 train / 1,693 test examples
- **Building On task**: ~2,843 train / ~1,739 test examples (fewer problems have prerequisite labels)
- Each problem maps to 1-5 CCSS codes (e.g., `3.OA.A.1`, `K.CC.A.1`, `A-SSE.A.1a`)

### CCSS Hierarchy

Standards have 4 levels of granularity:
- **Standard** (most specific): `3.OA.A.1` — the full code
- **Cluster**: `3.OA.A` — groups related standards
- **Domain**: `3.OA` — broad content area within a grade
- **Grade**: `3` — grade level (K, 1-8, HS)

We evaluate at all 4 levels. Coarser levels are more forgiving — a prediction of `3.OA.A.2` when the answer is `3.OA.A.1` fails at standard level but succeeds at cluster, domain, and grade.

## Method

### Fine-Tuning Setup

- **Technique**: QLoRA (4-bit quantized Low-Rank Adaptation) via Unsloth
- **LoRA config**: r=16, alpha=16, targeting q/k/v/o/gate/up/down projections
- **Training**: SFTTrainer from TRL, batch_size=8 (1B/3B) or 4 (8B), gradient accumulation 4-8 steps, lr=2e-4, AdamW 8-bit
- **Hardware**: NVIDIA A100-SXM4-40GB/80GB on Google Colab Pro
- **Evaluation**: 200 random test problems (seed=42) for fast iteration

### Prompt Format

System prompt instructs the model to output CCSS codes as a comma-separated list. Input is the raw math problem text. Target is the ground truth standard codes for the relevant relation type (Addressing or Building On).

### Models Tested

| Model | Params | Family | Epochs | Task |
|-------|--------|--------|--------|------|
| Llama 3.2 1B Instruct | 1.23B | Llama 3.2 | 3, 8 | Addressing |
| Llama 3.2 3B Instruct | 3.21B | Llama 3.2 | 3 | Addressing |
| Llama 3.1 8B Instruct | 8B | Llama 3.1 | 3 | Addressing + Building On |
| Llama 3.2 1B Instruct | 1.23B | Llama 3.2 | 8 | Building On |

### Performance Enhancements Tried

1. **More training epochs** (3 → 8): 2x F1 improvement at standard level for 1B. Model hadn't converged at 3 epochs.
2. **Larger model scale** (1B → 3B → 8B): 3B at 3 epochs beat 1B at 8 epochs. 8B at 3 epochs doubled the 3B. Scale matters more than training duration.
3. **Task-specific training** (Addressing vs Building On): Separate models for each relation type rather than multi-task.

## Results Summary

### Addressing Task — F1 by Granularity

| Model | Standard | Cluster | Domain | Grade |
|-------|----------|---------|--------|-------|
| 1B base (zero-shot) | 0.008 | 0.013 | 0.028 | — |
| 1B fine-tuned (3 ep) | 0.059 | 0.114 | 0.224 | — |
| 1B fine-tuned (8 ep) | 0.119 | 0.218 | 0.306 | 0.493 |
| 3B fine-tuned (3 ep) | 0.154 | 0.256 | 0.342 | 0.543 |
| 8B fine-tuned (3 ep) | 0.326 | 0.457 | 0.552 | 0.539 |

### Building On Task — F1 by Granularity

| Model | Standard | Cluster | Domain | Grade |
|-------|----------|---------|--------|-------|
| 1B fine-tuned (8 ep) | 0.071 | 0.139 | 0.207 | 0.373 |
| 8B fine-tuned (3 ep) | 0.301 | 0.369 | 0.420 | 0.539 |

Building On is consistently harder than Addressing — prerequisite standards are less directly tied to the problem text.

### A2 Zero-Shot Baselines (for comparison)

| Model | Standard F1 | Domain F1 |
|-------|-------------|-----------|
| GPT-5.2 (zero-shot) | 0.500 | 0.643 |
| Claude Sonnet 4.6 (zero-shot) | 0.477 | 0.581 |
| Gemini (zero-shot) | 0.455 | 0.508 |

Note: A2 baselines used a different 40-problem validation set with in-context examples and full CCSS descriptions in the prompt. Not directly comparable but directionally useful. The 8B fine-tuned model (standard F1=0.326) is approaching frontier zero-shot performance despite being 100x+ smaller.

## Evaluation Metrics

- **Precision**: Of the standards the model predicted, what fraction were correct?
- **Recall**: Of the ground-truth standards, what fraction did the model find?
- **F1**: Harmonic mean of precision and recall
- **Exact Match**: Fraction of problems where the predicted set exactly matches ground truth

All metrics are computed per-problem and averaged across the eval set.

## Evals Folder Guide

The `evals/` folder contains all raw evaluation data as JSON files.

### File Naming Convention

```
evaluation_results_{size}_{task}.json    — standard/cluster/domain metrics
grade_level_{size}_{task}.json           — grade-level + per-grade breakdown
```

Where:
- `{size}` = `1b`, `3b`, or `8b`
- `{task}` = `assess` (Addressing) or `building_on` (Building On / Prerequisites)

### File Contents

**`evaluation_results_*.json`** contains:
- `base_model.{standard,cluster,domain}` — pre-fine-tune metrics (precision, recall, f1, exact_match)
- `finetuned_model.{standard,cluster,domain}` — post-fine-tune metrics
- `num_test` — total test set size (1,693)
- `num_train` — training set size
- `num_eval` — eval subset size (200) (in Building On files)

**`grade_level_*.json`** contains:
- `overall_grade_level.{base,finetuned}` — aggregate grade-level P/R/F1/EM
- `finetuned_per_grade` — breakdown by grade (K, 1-8, HS) with `n` (count), `std_f1`, `domain_f1`, `grade_f1`

### Current Files

| File | Model | Task | Contents |
|------|-------|------|----------|
| `evaluation_results 1b_assess.json` | 1B (8 ep) | Addressing | std/cluster/domain |
| `evaluation_results_3b_assess.json` | 3B (3 ep) | Addressing | std/cluster/domain |
| `evaluation_results_8b_assess.json` | 8B (3 ep) | Addressing | std/cluster/domain |
| `evaluation_results_1b_building_on.json` | 1B (8 ep) | Building On | std/cluster/domain |
| `evaluation_results_8b_building_on.json` | 8B (3 ep) | Building On | std/cluster/domain |
| `grade_level_1b_assess.json` | 1B (8 ep) | Addressing | grade + per-grade |
| `grade_level_3b_assess.json` | 3B (3 ep) | Addressing | grade + per-grade |
| `grade_level_8b_assess.json` | 8B (3 ep) | Addressing | (pending) |
| `grade_level_1b_building_on.json` | 1B (8 ep) | Building On | grade + per-grade |
| `grade_level_8b_building_on.json` | 8B (3 ep) | Building On | grade + per-grade |

## Annotation Disagreement Analysis

(See below — preserved from A2 analysis.)

---

## Current IRR
- Standard-level alpha: **0.263** (well below 0.667 threshold)
- Agreement improves at coarser levels (domain: 0.413, grade: 0.492) but still marginal everywhere

## Key Disagreement Patterns

### 1. Grade-Level Drift
- Annotators place the same problem **2-4 grade levels apart**
- `im_lesson_005434`: Krish=3.OA, Sera=7.EE, Zane=7.NS, Teacher=6.NS
- `im_task_000032`: Krish=7.NS (after initially skipping), Sera/Zane=1.OA, Teacher=2.OA
- `im_center_000001`: Students=4.NBT.A.3, Teacher=K.NBT.A.1 (4 grade levels apart)
- **Cause**: No protocol for anchoring to the problem's intended grade level before selecting standards. Annotators rely on gut feel about mathematical content rather than what the problem is actually assessing.

### 2. Teacher Buddy Labels Prerequisites, Students Label Target Skill
- Teacher systematically picks foundational/prerequisite standards ("Building On") where students pick the assessed standard ("Addressing")
- `im_task_000484` (snail rate problem): Students=6.RP.A.2/A.3b, Teacher=4.OA.A.2/B.4
- `im_lesson_002158`: Krish/Sera=3.OA.B.6/A.4, Teacher=4.OA.A.1/B.4
- `im_lesson_005682`: Krish/Zane=F-IF.A.2, Teacher=6.EE.B.5/C.9
- **Cause**: Teacher thinks pedagogically ("what do students need to know first?") vs. students think analytically ("what is this problem testing?"). The codebook defines Addressing/Building On/Building Towards but doesn't give a decision rule for which to prioritize.

### 3. Parent vs. Child Standard Inconsistency
- No rule on granularity -- some label 7.NS.A.1 (parent), others label 7.NS.A.1c + 7.NS.A.1d (children)
- `fl_problem_001090`: Zane/Teacher label parent 7.NS.A.1; Krish labels only children 7.NS.A.1c/1d
- `im_lesson_003688`: All in 2.NBT but scattered across A.1a, A.4, B.5, B.9
- **Cause**: No guideline on when parent vs. child is appropriate. Annotators make ad hoc decisions.

### 4. Label Count Varies Wildly (1 to 5 per problem)
- `fl_problem_001090`: Sera=1 standard, Zane=3 standards
- `im_lesson_004151`: Zane=1, Krish=3, Sera=2, Teacher=2
- **Cause**: No target range for how many "Addressing" labels to assign. Some annotators are conservative (pick the single best), others are expansive.

### 5. Sera Outliers
- `im_lesson_008687`: Sera=1.OA.B.3/1.OA.D.7 (grade 1), everyone else=grade 6-7 EE domain
- `im_lesson_005682`: Sera=F-TF.A.2/A.3 (trig functions), Krish/Zane=F-IF.A.2 (function notation) -- wrong F- subdomain
- `fl_problem_001090`: Sera labeled 6.NS.C.7 on first pass, then skipped on second pass
- **Cause**: Likely insufficient familiarity with the CCSS hierarchy at higher grade levels. The annotation tool's drill-down UI may have led to picking visually similar but semantically wrong standards.

### 6. Domain Confusion Within Grade Bands
- **OA vs. NBT** (grades 1-4): `im_lesson_004226` -- Krish/Zane=K.OA, Sera/Teacher=1.OA
- **NS vs. NF** (grades 4-6): `im_task_000397` -- Krish/Zane=6.NS.A.1, Sera/Teacher=4.NF.B.3
- **EE vs. F** (grades 6-HS): `im_lesson_005682` -- Teacher=6.EE, Krish/Zane=F-IF, Sera=F-TF
- **Cause**: These domain pairs share overlapping vocabulary (e.g., both OA and NBT involve whole-number computation). Without a decision tree, annotators default to whichever domain they encounter first in the tool.

### 7. Krish Revision Artifacts
- Duplicate entries for `fl_problem_001778`, `im_lesson_003688`, `im_task_000397`, `im_task_000032`
- First entry for `fl_problem_001778` was F-TF.C.8 (Pythagorean identity) -- completely wrong domain for a polynomial problem
- **Cause**: Annotation tool allowed re-submission without clearing previous entry. Also suggests uncertainty during initial labeling.

## Where Agreement Was Strong
- `im_lesson_004984` (bar graphs): All 4 picked **2.MD.D.10** -- perfect match
- `fl_problem_000707`: All 4 skipped -- unanimous
- `im_lesson_007377`: 3/4 on **3.NBT.A.2** (Sera picked 3.OA.D.9 -- same grade, different domain)
- `im_lesson_009583`: 3/4 on **F-TF.A.2** (Sera picked F-TF.A.3 -- neighbor standard)
- `fl_problem_001778`: All eventually converge on **A-SSE.A.2** as one of their labels
- **Pattern**: Agreement is highest when (a) the problem clearly maps to a single unambiguous standard, or (b) the domain is narrow/specialized (trig, data/graphs)

## Proposed Protocol Rules to Increase IRR

### Rule 1: Grade-First Anchoring
- Before selecting any standard, annotator must **independently determine the grade band** (K-2, 3-5, 6-8, HS) based on the mathematical content
- If the problem has metadata indicating its source grade level, use that as a starting anchor
- This alone would have prevented the worst disagreements (e.g., grade 1 vs. grade 7)

### Rule 2: Strict "Addressing" Definition
- **Addressing** = "If this were a test question, what standard is the teacher grading?" Pick ONLY the directly assessed skill
- **Building On** = prerequisite knowledge assumed but not tested
- Default to Addressing unless explicitly annotating prerequisites
- Resolves the teacher-buddy vs. student split

### Rule 3: Most-Specific Standard
- Always label at the **most specific (child) level** available
- Only use parent standard if no single child captures the problem
- E.g., use 7.NS.A.1d, not 7.NS.A.1, if the problem specifically tests "apply properties of operations to add/subtract rationals"

### Rule 4: Target 1-2 Addressing Labels
- Assign **1-2 Addressing standards** per problem (hard cap at 3)
- Forces annotators to commit to the primary skill rather than hedging with 5 labels

### Rule 5: Domain Decision Trees for Common Confusions
- **OA vs. NBT**: Is the focus on the operation/relationship (OA) or on place-value mechanics (NBT)?
- **NS vs. NF**: Is the problem about fraction concepts specifically (NF) or broader number system operations (NS)?
- **EE vs. F**: Is there an explicit input-output/function relationship (F) or just equation manipulation (EE)?

### Rule 6: Mandatory Read-Back
- After selecting a standard, annotator must **read the full standard description** and confirm it matches the problem
- Prevents picking a neighbor standard based on a similar-sounding name

### Rule 7: Calibration Rounds
- Before independent annotation, do **5 problems together** as a group with discussion
- Document edge-case decisions as precedents (e.g., "rate problems with whole-number division: 6.RP not 4.OA")
- Re-calibrate every 10-15 problems

### Rule 8: No Silent Skips or Re-submissions
- If skipping, annotator must write a 1-sentence reason
- Annotation tool should prevent duplicate submissions (or flag them for resolution)
