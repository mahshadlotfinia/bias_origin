# Race and gender differences in medical imaging models arise at the shared decision threshold

This is the official repository of the paper [Race and gender differences in medical imaging models arise at the shared decision threshold](https://arxiv.org/abs/2608.25981).

## Overview

The code compares the performance of medical imaging models between patient groups. Each model is a logistic regression classifier on the frozen features of one image encoder for one finding, with one decision threshold for all groups. For each model and group attribute, the code computes the difference between the largest and the smallest group value of the area under the receiver operating characteristic curve (AUROC), the sensitivity, and the false positive rate (FPR). It tests this difference against its distribution under equal group performance, which comes from resampling patients with all of their images. The tests run over all test patients, within each hospital, and within the radiograph views and age bands of a hospital.

## Installation

```bash
git clone https://github.com/mahshadlotfinia/bias_origin.git
cd bias_origin
```

Install PyTorch, torchvision, timm, Transformers, huggingface_hub, OpenCLIP (`open_clip_torch`), TorchXRayVision, NumPy, SciPy, pandas, scikit-learn, Pillow, PyYAML, and tqdm. Install FLAIR from its [repository](https://github.com/jusiro/FLAIR).

## Configuration

Every path is read from `config/config.yaml` by `config/serde.py`, and so is every setting of the analyses except the constants at the top of a module. Set the roots at the top of `config/config.yaml`: `ocean_root` for the outputs, `home_root` for the folder that holds `Repositories/bias_origin`, `user_home` for the Hugging Face cache, and `datasets_root` for the datasets. Each analysis has a `main_` function that takes the path of `config/config.yaml`.

For each chest radiograph site under `cxr.sites`, `master_csv` names a master list with one row per image. Each row identifies the image and gives its labels, its split, and its metadata. The column names of each site are set in the same block. The dermatology and fundus sources are set under `derm.sources` and `fundus.sources`. The compared attributes, the strata, and the output folders of the analyses are set under `subgroups`. Enter a Hugging Face token in `hf_token` to load the gated checkpoints. With `hf_token: null`, only the ungated checkpoints load.

## Encoder panel

The name in parentheses is the key of each encoder under `encoder_panel.image`.

- RAD-DINO (`rad_dino`): `microsoft/rad-dino`
- BiomedCLIP (`biomedclip_image`): `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- TorchXRayVision DenseNet-121 (`txrv_densenet`): the `densenet121-res224-all` weights of the `torchxrayvision` package
- DINOv3 ViT-L, ViT-B, and ViT-S (`dinov3_l`, `dinov3_b`, `dinov3_s`): `facebook/dinov3-vitl16-pretrain-lvd1689m`, `facebook/dinov3-vitb16-pretrain-lvd1689m`, `facebook/dinov3-vits16-pretrain-lvd1689m`
- DINOv2 ViT-L (`dinov2_large`): `facebook/dinov2-large`
- CLIP ViT-L/14 (`clip_vitl14`): `openai/clip-vit-large-patch14`
- SigLIP 2 ViT-L (`siglip2_large`): `google/siglip2-large-patch16-512`
- MONET (`monet`): `chanwkim/monet`
- DermLIP ViT-B/16 (`panderm`): `redlessone/DermLIP_ViT-B-16`
- RETFound (`retfound`): `iszt/RETFound_mae_meh`
- FLAIR (`flair`): the package from https://github.com/jusiro/FLAIR
- A randomly initialized ViT-S/16 (`random_init_vit_s`): `vit_small_patch16_224` in timm

The DINOv3 checkpoints need an accepted license on Hugging Face, and DermLIP and RETFound need accepted terms there. The other checkpoints download without an access request. The encoders pretrained in this study start from the DINOv3 ViT-S and ViT-B checkpoints.

## Code structure

- `data_loader/`: the chest radiograph pool (`build_cxr_pool.py`) with the race and insurance of MIMIC-CXR from MIMIC-IV (`build_mimic_demographics.py`), the dermatology pool of ISIC 2019, Fitzpatrick17k, and the Diverse Dermatology Images set (`build_derm_pool.py`), and the fundus pool (`build_fairvision_pool.py`). The labels and the group attributes are harmonized in `cxr_harmonization.py` and `sensitive_harmonization.py`. `preprocess_manifest` in `preprocess_utils.py` writes the stored copies of the images. `build_cxr_paired_reports.py` pairs each pretraining radiograph with its report and builds the two edited reports. `build_published_claims.py` holds the group values and group sizes of the published studies, transcribed from the articles.
- `encoders/`: each encoder of the panel with its preprocessing (`image_encoders.py`), the extraction and caching of the features (`extract_embeddings.py`), and the extraction of the features of the encoders pretrained in this study (`extract_controlled_embeddings.py`).
- `controlled/`: the natural and race-balanced pretraining sets (`build_training_mixtures.py`) and the pretraining of the encoders of this study (`train_encoder.py`). `run_one` trains one encoder.
- `analysis/`: the classifier and its threshold (`heads.py`), the resampling of patients, the group difference, and its test (`patient_bootstrap.py`), and the difference expected by chance from group sizes with the Hanley-McNeil variance or the binomial distribution (`count_chance.py`).
- `experiments/`: one module per analysis.
  - `scores.py` fits the classifier of every encoder and finding and saves its threshold and its test scores.
  - `chance.py` computes the group values, the group differences, their resampled distributions, and their tests over all test patients and within each hospital and view, and the tests of the mean difference within the strata of hospital, view, and age band.
  - `validation.py` tests permuted race groups, adds noise to the scores of one group, and simulates studies for the form of the test that uses group sizes.
  - `contrasts.py` compares paired pretraining runs and computes how much each group difference varies over the seeds of a pretraining condition.
  - `published.py` tests the group differences of the published studies from their group values and group sizes, with the Hanley-McNeil variance and with the variance bound of Birnbaum and Klose for AUROC.
  - `summary_tables.py` summarizes the tests over the models, the order of the groups at the shared threshold, the group sizes, and the published and validation results.
  - `reported_statistics.py` computes the other summaries that the paper reports: the critical values of the tests, the counts by encoder and by pretraining condition, the tests without the encoders whose pretraining data include test radiographs, the comparison of resampled patients with image counts, the paired comparisons and their variation over seeds, the age of the insurance groups, the dataset counts, and, given the released list of pretraining images of RAD-DINO, the overlap of that list with the test images.
- `Inference/`: the resampling and the false discovery rate correction (`stats_utils.py`), and the result rows and the resumption of interrupted runs (`report_utils.py`).
- `config/`: the configuration and its reader.

Every analysis saves its results per unit of work, skips the units that are already finished, and resumes an interrupted run.

## Data

The chest radiographs come from MIMIC-CXR (with demographics from MIMIC-IV), CheXpert (with demographics from CheXpert Plus), NIH ChestX-ray14, PadChest, VinDr-CXR, and VinDr-PCXR. The dermatology images come from ISIC 2019, Fitzpatrick17k, and the Diverse Dermatology Images set, and the fundus images come from Harvard-FairVision. This repository redistributes none of them. The manifests are not included either, since they contain patient identifiers and demographics from credentialed datasets. Obtain each dataset from its source, then build the manifests with the `main_build_` functions in `data_loader/`. Every dataset and every model comes with a license and terms of use. Following them is the responsibility of the user.

## Citation

```bibtex
@article{lotfinia2026race,
  title   = {Race and gender differences in medical imaging models arise at the shared decision threshold},
  author  = {Lotfinia, Mahshad and Truhn, Daniel and Maier, Andreas and Tayebi Arasteh, Soroosh},
  journal = {arXiv preprint arXiv:2608.25981},
  year    = {2026}
}
```

## License

MIT. See `LICENSE`.
