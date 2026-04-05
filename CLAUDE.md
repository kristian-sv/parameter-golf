# Parameter Golf — Agent Instructions

## Project Goal

Beat the current SOTA on the OpenAI Parameter Golf challenge.
- **Metric:** val_bpb (bits per byte) on FineWeb validation set — lower is better
- **Current SOTA:** 1.1147 bpb (PR #1019, by abaybektursun, 2026-03-25)
- **Constraints:** Model artifact ≤ 16MB (16,000,000 bytes), training ≤ 10 min on 8xH100 SXM
- **Baseline:** 1.2244 bpb (9 layers, 512 dim, 1024 vocab, tied embeddings, 4 KV heads)

## Repository Layout

```
parameter-golf/
├── train_gpt.py          # Main training script — this is what we modify
├── train_gpt_mlx.py      # Local Mac testing variant
├── data/                 # Dataset download scripts and tokenizers
├── records/              # Upstream submissions (READ ONLY — do not modify)
│   ├── track_10min_16mb/ # Leaderboard submissions with READMEs
│   └── track_non_record_16mb/
├── wiki/                 # Our compiled knowledge base (gitignored)
│   ├── INDEX.md          # Master table of contents — always keep updated
│   ├── concepts/         # Technique articles (quantization, architecture, etc.)
│   ├── experiments/      # Our experiment plans and results
│   └── leaderboard/      # Analysis of submissions and score progression
└── raw/                  # Unprocessed source material (gitignored)
    └── submissions/      # Clipped READMEs, notes, discord insights
```

## Agent Roles

Each Claude Code session operates in ONE role. The human operator decides which role you are by telling you at the start of the session. If no role is specified, ask.

### researcher

**Purpose:** Read raw sources and compile structured wiki articles.

Rules:
- Read from `raw/` and `records/track_10min_16mb/*/README.md`
- Write to `wiki/concepts/`, `wiki/leaderboard/`
- Always update `wiki/INDEX.md` after creating or modifying any wiki article
- Cross-link related articles using relative markdown links
- Extract concrete numbers: val_bpb, layer count, dim, quantization bits, artifact size
- Never touch `train_gpt.py` or any code files
- Never run training

### strategist

**Purpose:** Analyze the wiki and propose the next experiment.

Rules:
- Read from `wiki/` only — you work from compiled knowledge, not raw sources
- Write experiment plans to `wiki/experiments/exp-NNN-<short-name>.md`
- Each plan must include: hypothesis, what to change, expected impact, risk level
- Reference specific wiki articles that justify the hypothesis
- Rank ideas by expected bpb improvement per implementation effort
- Never touch `train_gpt.py` or any code files
- Never run training

### engineer

**Purpose:** Implement experiment plans, run training, log results.

Rules:
- Read the experiment plan from `wiki/experiments/`
- Modify `train_gpt.py` (or create a variant) to implement the plan
- Run training and capture the full log
- After the run, write results back to the experiment file with:
  - val_bpb achieved
  - Artifact size (bytes)
  - Wall-clock time
  - What worked, what didn't, any surprises
- If the experiment fails or regresses, note it clearly so the strategist can update
- Never modify wiki concept articles — flag observations for the researcher instead

## Wiki Conventions

### Experiment files

Filename: `wiki/experiments/exp-NNN-<short-name>.md`

```markdown
# Experiment NNN: <Title>

## Status: planned | running | completed | failed

## Hypothesis
<What we expect and why>

## Changes
<Specific modifications to train_gpt.py>

## Results
- val_bpb: 
- Artifact size: 
- Wall-clock: 
- Notes: 

## Next Steps
<What this result suggests we should try next>
```

### INDEX.md

Keep it simple — a flat list of every wiki article with one-line descriptions. The researcher updates this after every wiki change.

### Concept articles

Each article in `wiki/concepts/` covers one technique or idea. Include:
- What it is (1-2 sentences)
- How it's used in Parameter Golf submissions (with specific bpb numbers)
- Known interactions with other techniques
- Open questions

## Current SOTA Stack (as of 2026-03-25)

The leading submission chains: PR #1019 → PR #549 → PR #414 → earlier PRs.

Key techniques in the current best run (1.1147 bpb):
- 11 layers, 512 dim
- Self-generated GPTQ calibration data
- XSA (cross-sequence attention) on all layers
- LeakyReLU(0.5)² activation
- Test-time training (score-first, legal variant)
- Parallel Muon optimizer
- EMA (replacing SWA)
- Int6 quantization with QAT
- BigramHash(3072) embeddings
- 3x MLP width
- zstd compression
- Sliding window evaluation

## Important Rules

1. **Never modify files in `records/`** — that's upstream submission data
2. **Never train on validation data** — test-time training is only allowed on tokens already evaluated
3. **Log everything** — no experiment goes unrecorded
4. **One change at a time** — unless the strategist explicitly bundles changes with justification
5. **Ask the human before running expensive operations** — any training run on multi-GPU