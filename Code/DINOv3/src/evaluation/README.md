# evaluation/

Evaluation, summarization, analysis, diagnostics, and figure-generation code for
the seismic denoising project.

The implementation is grouped by purpose:

| Folder | Purpose |
|---|---|
| `common/` | Shared metrics and path/bootstrap helpers |
| `evaluators/` | Per-checkpoint evaluation entrypoints |
| `summarizers/` | Cross-run aggregation and report generation |
| `analysis/` | Mechanism, attention, representation, and counterfactual diagnostics |
| `figures/` | Comparison panels, thesis/poster figures, and plotting scripts |
| `diagnostics/` | Raw-data audits, noisy-clean baselines, input-quality utilities |
| `timing/` | Inference and training-time measurement scripts |
| `maintenance/` | One-off layout or migration utilities |
| `runners/` | Local PowerShell wrappers for common Windows workflows |

Root-level `.py` files are compatibility wrappers. Existing commands such as
`python evaluation/evaluate.py ...` and
`python Code/DINOv3/src/evaluation/summarize_impeccable_runs.py ...` continue to
work, but new code should import from the grouped modules.

Preferred module form from `Code/DINOv3/src`:

```powershell
python -m evaluation.evaluators.evaluate --config configs/<config>.yaml --checkpoint <best.pt>
python -m evaluation.summarizers.summarize_impeccable_runs --project-root C:/UNI/Y3/RP
python -m evaluation.figures.generate_comparison_panels --project-root C:/UNI/Y3/RP
python -m evaluation.analysis.analyze_mechanism --project-root C:/UNI/Y3/RP
```

Legacy flat-path form remains valid for DAIC scripts and older docs:

```powershell
python evaluation/summarize_impeccable_runs.py --project-root C:/UNI/Y3/RP
```

For canonical run commands and output locations, see
`docs/research/operations/how_to_run_summarizers.md`.

## Evaluators

| Module | Legacy wrapper | Purpose |
|---|---|---|
| `evaluation.evaluators.evaluate` | `evaluation/evaluate.py` | Paired Image Impeccable evaluation |
| `evaluation.evaluators.evaluate_robustness` | `evaluation/evaluate_robustness.py` | F3 no-reference field-transfer evaluation |
| `evaluation.evaluators.evaluate_stitched` | `evaluation/evaluate_stitched.py` | Full-section overlap-stitching evaluation |
| `evaluation.evaluators.evaluate_filtered_reference` | `evaluation/evaluate_filtered_reference.py` | F3 filtered-reference agreement evaluation |

## Summarizers

| Module | Output family |
|---|---|
| `evaluation.summarizers.summarize_impeccable_runs` | Main, data-efficiency, and index summaries |
| `evaluation.summarizers.summarize_ablations` | Ablation reports |
| `evaluation.summarizers.summarize_robustness` | F3 robustness reports |
| `evaluation.summarizers.summarize_filtered_reference` | F3 filtered-reference summaries |
| `evaluation.summarizers.summarize_stitched` | Stitching diagnostics |
| `evaluation.summarizers.summarize_efficiency` | PEFT vs full-FT efficiency |
| `evaluation.summarizers.summarize_100train_channel_window` | 100-train channel-window pilot |
| `evaluation.summarizers.summarize_mechanism_controls` | Trained mechanism controls |

## Local Runners

PowerShell wrappers live in `evaluation/runners/`. Example:

```powershell
pwsh Code/DINOv3/src/evaluation/runners/summarize_impeccable_runs.ps1
```

