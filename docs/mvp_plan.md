# LensGraph Dev-Small MVP Plan (Execution Order)

## Phase 1 (Done)
- Catalog simulator scaffold with lensed + isolated events.
- PM/SIS/SIE-style sampler hooks.
- CLI generation script and schema tests.

## Phase 2 (Next)
1. Add `lensgraph/models/encoder.py` with PI-ResNet-compatible interface.
2. Add `lensgraph/models/losses.py` for supervised contrastive loss.
3. Add `scripts/02_train_encoder.py` and compute Recall@1/5/10 on validation split.

## Phase 3
1. Add FAISS-based ANN index (`retrieval/ann_index.py`).
2. Add candidate edge graph construction (`retrieval/candidate_graph.py`).
3. Add baseline exhaustive pair runner for timing reference.

## Phase 4
1. Add graph partition baseline (threshold + connected components).
2. Add system-level metrics module:
   - System Recall
   - Exact System Recovery
   - Isolation Specificity
   - Catalog FDR
3. Add `scripts/04_run_lensgraph.py` end-to-end runner.
