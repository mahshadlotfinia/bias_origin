# FRAME: separating sampling variation from representational cause in medical imaging fairness

## Overview

This is the official repository of the paper **FRAME: separating sampling variation from representational cause in medical imaging fairness**.

Medical imaging models are widely reported to perform differently across demographic subgroups, and the usual response is to remove the demographic information that a model encodes. This repository holds the code for Fair-model Reference And Mechanism Evaluation (FRAME), a two-step audit of such a report, and for the study that applies it. Step one derives the fair-model reference, which is the distribution that a reported subgroup difference takes under exact fairness at the subgroup sizes it was computed on. Step two tests whatever remains of the difference, in the representation that the scores were read from.

The pipeline measures a subgroup difference the same way throughout, on frozen features from a panel of image encoders with a light classification head, which is the common deployment setting. It then varies one factor at a time: the mitigation method, from preprocessing, in-processing, postprocessing, linear concept erasure, and optimal transport; the pretraining objective, the backbone scale, the demographic composition of the pretraining data, and the demographic content of the paired reports, through a matrix of encoders pretrained from scratch under otherwise identical conditions; how much of the backbone is unfrozen during finetuning; the modality, across chest radiographs, dermatology, and fundus photographs; and the technical acquisition variables that are known to differ across subgroups. Two controls check the measurement itself. A synthetic construction gives a case whose answer is known by design, and an injection adds a demographic effect of a known size to real cached features and re-runs the identical battery on them. The same reference is applied at the end to subgroup differences that other groups have already published, from each article's own reported values and subgroup counts.

## Encoder panel

The panel spans domain-specific medical encoders, general-purpose vision encoders at three scales, and an untrained control at matched scale, so a difference between encoders is not confounded with the size of the model. All are open-weight and loaded from public checkpoints. The DINOv3 checkpoints are gated and need an accepted license on Hugging Face plus a token, read from the config file.

| Encoder | Identifier | Dim | Pretraining objective | Domain |
|---|---|---|---|---|
| RAD-DINO | `microsoft/rad-dino` | 768 | Self-supervised | Chest radiograph |
| BiomedCLIP | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | 512 | Image-text | Chest radiograph |
| TorchXRayVision DenseNet | built into `torchxrayvision` | 1024 | Label-supervised | Chest radiograph |
| MONET | `chanwkim/monet` | 768 | Image-text | Dermatology |
| PanDerm | `redlessone/DermLIP_ViT-B-16` | 512 | Image-text | Dermatology |
| RETFound | `iszt/RETFound_mae_meh` | 1024 | Self-supervised | Fundus |
| FLAIR | local install from the upstream repository | 512 | Image-text | Fundus |
| DINOv3 ViT-S | `facebook/dinov3-vits16-pretrain-lvd1689m` | 384 | Self-supervised | General; gated |
| DINOv3 ViT-B | `facebook/dinov3-vitb16-pretrain-lvd1689m` | 768 | Self-supervised | General; gated |
| DINOv3 ViT-L | `facebook/dinov3-vitl16-pretrain-lvd1689m` | 1024 | Self-supervised | General; gated |
| DINOv2 Large | `facebook/dinov2-large` | 1024 | Self-supervised | General |
| CLIP ViT-L/14 | `openai/clip-vit-large-patch14` | 768 | Image-text | General |
| SigLIP 2 Large | `google/siglip2-large-patch16-512` | 1152 | Image-text | General |
| Randomly initialized ViT-S | `vit_small_patch16_224`, no weights | 384 | None | Control |

Alongside the panel, the pipeline pretrains its own encoders from scratch on chest radiographs, crossing the pretraining objective (self-supervised contrastive, label-supervised, and image-text with a clinical text tower), the backbone scale (ViT-S and ViT-B), and the training corpus (its natural composition, a race-balanced resample, and two variants of the paired reports in which demographic mentions are removed or added). Because everything else is held fixed, the difference between two of these encoders identifies the effect of the factor that changed.

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/mahshadlotfinia/bias_origin.git
```

Two packages are not on PyPI and are installed from their upstream repositories. FLAIR is needed for the fundus encoder of the same name, and `rlace` is needed only for the third erasure method, which is implemented and is left out of the enabled battery.

```bash
pip install git+https://github.com/jusiro/FLAIR.git
pip install git+https://github.com/shauli-ravfogel/rlace.git
```

### 2. Configuration

Every path and run option lives in one YAML file, `config/config.yaml`, resolved by `config/serde.py`. Machine-specific roots are declared once at the top and every other path is built from them by interpolation, so no path joining happens in application code. Moving the repository between machines means editing that block and the single `GLOBAL_CONFIG_PATH` constant in `main_bias_origin.py`, and nothing else.

```yaml
ocean_root:    /path/to/storage/Documents      # outputs live here
home_root:     /path/to/home/Documents         # code lives here
user_home:     /path/to/home
datasets_root: /path/to/datasets               # its own root, not derived

BiasOrigin:
  hf_token: "hf_YOUR_TOKEN_HERE"               # required for the gated checkpoints
  seed: 42

  cxr:
    sites:                                     # one block per source, with its own
      mimic: {image_root: ..., master_csv: ...}  # column names and path convention

  encoder_panel:
    image:                                     # the frozen panel above

  controlled:                                  # the encoders pretrained from scratch
    objectives:   [ssl, supervised, image_text]
    backbones:    [vit_s, vit_b]
    data_compositions: [natural, balanced]

  mitigation:
    methods_all_encoders: [resample, reweigh, group_dro, adversarial, reduction,
                           eo_shift, platt_recal, leace, inlp]
    attributes: [race_grp, age_grp]

  stats:
    n_boot: 1000                               # patient-clustered bootstrap redraws
    boot_seed: 0
    operating_sensitivity: 0.80
    fdr_alpha: 0.05
    bootstrap_n_jobs: auto                     # reads the job's CPU allocation
```

`mitigation.methods_all_encoders` is load-bearing and not decorative. It is the single source of truth for which methods run. A name removed from it stops running, and a name that no method implements raises at startup instead of being skipped in silence.

> **Third-party data and model compliance (your responsibility).** This repository redistributes no dataset and no model weights. It points at public sources and loads checkpoints you obtain yourself. Before using any dataset, model, or third-party service referenced here, YOU are responsible for reviewing and complying with its license, terms of use, data-use agreement, and any applicable privacy, ethics, or regulatory requirement for your jurisdiction and intended use. The authors make no representation that any particular use is permitted.

## Datasets

Chest radiographs are pooled from six public sources, with per-source label decoding, view filtering, and path resolution handled in one harmonization module so every site enters the pool on the same scale. Dermatology is pooled from three sources and fundus photographs from one. Demographic attributes are mapped onto one canonical vocabulary in a single module, which is the only place a subgroup definition exists.

Binary labels follow the CheXpert convention throughout: positive is 1, negative is 0 and 2, and 3 is excluded per finding and never recoded. It is the label policy for the whole repository and it is enforced in the harmonization layer, not at each call site.

Every stage is resumable and skips work that is already finished. Extraction writes one cache per encoder and pool and skips a complete one. Training writes a per-epoch checkpoint atomically, so an interrupted run resumes from its last finished epoch. The heavy analyses write one result shard per unit and save a partial result after every finding, so an interruption costs one finding and not a day. A merge step then unions the shards and applies the multiplicity correction once across the whole panel, which keeps the correction identical to a single-pass run.

**Running several jobs at once.** The sharded stages are the intended way to use a cluster. Submit the same script several times calling the same stages, and each job claims the next unclaimed unit instead of repeating work another job is already doing. A claim is a small file next to the shards, refreshed while its owner is alive, and reclaimed when the owner's job is gone. Give every CPU stage an explicit core allocation, since the controller pins the linear-algebra thread count to it and prints both the cores and the memory limit the job was given as its first two lines.

**Resume caveat.** Resume checks whether a unit's output exists and whether the parameters recorded inside it match what this run needs. It cannot detect an arbitrary code change. If you edit a stage, delete its caches, checkpoints, shards, partial saves, and consolidated output before re-running, or it will skip every unit and keep the old numbers.

## File overview

- `config/config.yaml`: Every path and run option. Machine-specific roots at the top, everything else built by interpolation.
- `config/serde.py`: Reads the config and resolves the `${var}` references.
- `main_bias_origin.py`: The single controller. Defines the config-path constant that every module imports, forces line-buffered output, raises the open-file limit, prints the job's memory and core allocation, pins the linear-algebra thread count to that allocation, and holds one wrapper per stage that imports its implementation only when it is called.
- `data_loader/`: Manifest construction. Per-site chest radiograph harmonization with the label policy and path resolution, the demographic linkage table, the three pool builders, the paired image-report builders with their scrubbed and amplified variants, the canonical demographic vocabulary, the embedding dataset classes, the optional image cache, the subgroup composition counts, and the transcribed table of the published subgroup differences that the reference is applied to.
- `encoders/`: Per-architecture embedding functions behind one dispatch, returning L2-normalized float32 features, plus the cache purge that frees each model's weights after its pass and the only module that writes an embedding cache.
- `analysis/`: Loading a cache and joining it to its manifest, fitting linear and MLP heads, and the fairness panel that computes every subgroup metric for one set of scores.
- `Inference/stats_utils.py`: The patient-clustered bootstrap, its paired form, permutation tests, and the multiplicity correction. The statistical constants live here and nowhere else.
- `Inference/report_utils.py`: The single reporting layer. Owns the two table schemas, enforces the metric-versus-statistic contract, and holds the shard, partial-save, and work-claim machinery every sharded stage uses.
- `controlled/`: Builds the pretraining corpora and trains the encoder matrix, one run per call, with per-epoch atomic checkpoints and offline-first model loading under a hard timeout.
- `mitigation/`: The intervention battery: reweighing and resampling, group-distributionally-robust and adversarial training and the reduction approach, per-group calibration and per-group threshold selection, linear concept erasure with a nonlinear classifier fitted afterwards, group-aware optimal transport, and the routine that finds the smallest subgroup difference reachable at matched disease performance.
- `mechanism/`: Linear and nonlinear demographic decodability, the geometric overlap between the disease and demographic directions, the collateral disease-performance cost of erasure, and the cross-fit predictor built on them.
- `experiments/`: One module per analysis, each sharded and resumable, each writing its own pair of result CSVs: the subgroup panel across the controlled encoders, the intervention battery and the frontier it reaches, the mechanism analysis and its transfer test, the cross-modality and cross-modal analyses, the finetuning dose-response, the synthetic construction, the fair-model reference for every reported difference, the injected-effect control, the acquisition control, and the reference applied to the published claims.
- `theory/proposition.py`: The synthetic study in which the geometric relationship between the disease and demographic directions is varied by construction, so the measurement can be checked where the answer is known.
- `figures/`: The figure scripts, one per display item, with the shared style module and the layout checker they are verified with.
- `environment/`: The pinned environment.

## Citation

If you use this repository, please cite our paper:

```bibtex
@article{lotfinia2026frame,
  title   = {FRAME: separating sampling variation from representational cause in medical imaging fairness},
  author  = {Lotfinia, Mahshad and Truhn, Daniel and Maier, Andreas and Tayebi Arasteh, Soroosh},
  journal = {[journal to be added]},
  year    = {2026}
}
```

## License

MIT License. See `LICENSE` for details.
