# FRAME: Separating sampling variation from performance disparities in medical image analysis


## Overview

A subgroup audit of a medical imaging model usually reports the largest minus the smallest subgroup value of a performance measure and compares it with zero. A perfectly fair model also produces a positive difference, and that difference grows as the subgroups become smaller. Fair-model Reference And Mechanism Evaluation (FRAME) audits a reported subgroup difference in two steps. The first step computes the fair-model reference, which is the difference expected from a perfectly fair model at the same subgroup counts. It needs only the overall performance and the number of cases in each subgroup. It therefore also applies to a difference printed in a published article. The second step tests candidate causes of the part of the difference above the reference. It either injects demographic signal into the cached features of an encoder or removes disease signal from some subgroups. It then refits the disease head and measures the change in the difference.

## Encoder panel

- RAD-DINO: `microsoft/rad-dino`
- BiomedCLIP: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- TorchXRayVision DenseNet-121: the `densenet121-res224-all` weights of the `torchxrayvision` package
- DINOv3 ViT-S, ViT-B, and ViT-L: `facebook/dinov3-vits16-pretrain-lvd1689m`, `facebook/dinov3-vitb16-pretrain-lvd1689m`, `facebook/dinov3-vitl16-pretrain-lvd1689m`
- DINOv2 ViT-L: `facebook/dinov2-large`
- CLIP ViT-L/14: `openai/clip-vit-large-patch14`
- SigLIP2-L: `google/siglip2-large-patch16-512`
- MONET: `chanwkim/monet`
- DermLIP ViT-B/16: `redlessone/DermLIP_ViT-B-16`
- RETFound: `iszt/RETFound_mae_meh`
- FLAIR: the package from https://github.com/jusiro/FLAIR
- An untrained ViT-S/16 (`vit_small_patch16_224` in timm) with random weights

The DINOv3 checkpoints need an accepted license on Hugging Face, and DermLIP and RETFound need accepted terms there. The other checkpoints download without an access request. The controlled encoders start from the DINOv3 ViT-S and ViT-B checkpoints.

## Installing

```bash
git clone https://github.com/mahshadlotfinia/bias_origin.git
cd bias_origin
conda env create bias_origin
conda activate bias_origin
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
pip install git+https://github.com/jusiro/FLAIR.git
```

## Configuration

Every path and every setting is in `config/config.yaml`, read by `config/serde.py`. The paths in it are built from four roots at the top of the file: `ocean_root`, `home_root`, `user_home`, and `datasets_root`. To run on another machine, set these four roots and `GLOBAL_CONFIG_PATH` in `main_bias_origin.py`. For each chest radiograph site under `cxr.sites`, `master_csv` names a master list, a CSV with one row per image. Each row identifies the image and lists its labels, its split, and its metadata. The column names of each site are set in the same block. Put your Hugging Face token in `hf_token` to load the gated checkpoints. With `hf_token: null`, only the ungated checkpoints load. Only the mitigation methods listed under `mitigation.methods_all_encoders` run.

## Running the pipeline

Each stage is one `main_*` function in `main_bias_origin.py`. It takes the path to the configuration file as its `cfg_path` argument, which defaults to `GLOBAL_CONFIG_PATH`. There is no command-line interface. Call the stages from a Python session or from a job script, and give each long stage a separate job.

```python
from main_bias_origin import main_build_cxr_pool, main_extract_image_embeddings, main_build_final_tables

main_build_cxr_pool()
main_extract_image_embeddings(encoder_names=["rad_dino"])
main_build_final_tables()
```

## Data

The chest radiographs come from MIMIC-CXR (with demographics from MIMIC-IV), CheXpert (with demographics from CheXpert Plus), NIH ChestX-ray14, PadChest, VinDr-CXR, and VinDr-PCXR. The dermatology images come from ISIC 2019, Fitzpatrick17k, and the Diverse Dermatology Images set, and the fundus images come from Harvard-FairVision. Each dataset is available from its source under its license. This repository redistributes none of them. The manifests are not included either, since they contain patient identifiers and demographics from credentialed datasets. Obtain each dataset from its source, then build the manifests with the `main_build_*` stages. Every dataset and every model comes with a license and terms of use. Following them is the responsibility of the user.

## Files

- `main_bias_origin.py`: the stage functions and the configuration path.
- `config/`: the configuration file and its reader.
- `data_loader/`: the pool manifests, the label and attribute harmonization, the paired reports, the subgroup counts, and the table of published subgroup differences.
- `encoders/`: feature extraction for every encoder in the panel.
- `controlled/`: the pretraining sets and the training of the controlled encoders.
- `analysis/`: the disease heads and the subgroup measures.
- `mitigation/`: the nine mitigation methods, the optimal transport method, and the achievable difference at matched disease performance.
- `mechanism/`: demographic decodability, the geometric overlap, and the erasure cost.
- `theory/`: the synthetic model.
- `experiments/`: one module per analysis.
- `Inference/`: the bootstrap, the permutation tests, the multiplicity correction, and the result tables.

## Citation

```bibtex
@article{lotfinia2026frame,
  title   = {FRAME: Separating sampling variation from performance disparities in medical image analysis},
  author  = {Lotfinia, Mahshad and Truhn, Daniel and Maier, Andreas and Tayebi Arasteh, Soroosh},
  journal = {arXiv preprint arXiv:2608.25981},
  year    = {2026}
}
```

## License

MIT. See `LICENSE`.
