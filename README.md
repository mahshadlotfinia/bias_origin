# FRAME: Separating sampling variation from performance disparities in medical image analysis


## Installation

```bash
git clone https://github.com/mahshadlotfinia/bias_origin.git
cd bias_origin
```

Install PyTorch, torchvision, timm, Transformers, huggingface_hub, OpenCLIP (`open_clip_torch`), TorchXRayVision, NumPy, SciPy, pandas, scikit-learn, Fairlearn, concept-erasure, Pillow, PyYAML, and tqdm. Install FLAIR from its. Fairlearn provides the exponentiated-gradient reduction, and concept-erasure provides LEACE. If either package is missing, the code skips the method that needs it, so check that both are installed.

## Configuration

Every path and every setting is read from `config/config.yaml` by `config/serde.py`. Set the roots at the top of the file: `ocean_root` for the outputs, `home_root` for the folder that holds `Repositories/bias_origin`, `user_home` for the Hugging Face cache, and `datasets_root` for the datasets. Each analysis has a `main_` function that takes the path of `config/config.yaml`.


## Encoder panel

The name in parentheses is the key of each encoder under `encoder_panel.image`.

- RAD-DINO (`rad_dino`): `microsoft/rad-dino`
- BiomedCLIP (`biomedclip_image`): `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- TorchXRayVision DenseNet-121 (`txrv_densenet`): the `densenet121-res224-all` weights of the `torchxrayvision` package
- DINOv3 ViT-L, ViT-B, and ViT-S (`dinov3_l`, `dinov3_b`, `dinov3_s`): `facebook/dinov3-vitl16-pretrain-lvd1689m`, `facebook/dinov3-vitb16-pretrain-lvd1689m`, `facebook/dinov3-vits16-pretrain-lvd1689m`
- DINOv2 ViT-L (`dinov2_large`): `facebook/dinov2-large`
- CLIP ViT-L/14 (`clip_vitl14`): `openai/clip-vit-large-patch14`
- SigLIP2-L (`siglip2_large`): `google/siglip2-large-patch16-512`
- MONET (`monet`): `chanwkim/monet`
- DermLIP ViT-B/16 (`panderm`): `redlessone/DermLIP_ViT-B-16`
- RETFound (`retfound`): `iszt/RETFound_mae_meh`
- FLAIR (`flair`): the package from https://github.com/jusiro/FLAIR
- An untrained ViT-S/16 (`random_init_vit_s`): `vit_small_patch16_224` in timm with random weights

The DINOv3 checkpoints need an accepted license on Hugging Face, and DermLIP and RETFound need accepted terms there. The other checkpoints download without an access request. The controlled encoders start from the DINOv3 ViT-S and ViT-B checkpoints.

## Code structure

- `data_loader/`: the chest radiograph pool (`build_cxr_pool.py`) with the race and insurance of MIMIC-CXR from MIMIC-IV (`build_mimic_demographics.py`), the dermatology pool of ISIC 2019, Fitzpatrick17k, and the Diverse Dermatology Images set (`build_derm_pool.py`), and the fundus pool (`build_fairvision_pool.py`). The labels and the sensitive attributes are harmonized in `cxr_harmonization.py` and `sensitive_harmonization.py`. `resize_tree` and `preprocess_manifest` in `preprocess_utils.py` write the stored 224-pixel copies of the images, `build_cxr_paired_reports.py` pairs each pretraining radiograph with its report and builds the two report variants, and `build_subgroup_counts.py` counts the positive and negative images of every subgroup. `build_published_claims.py` holds the published subgroup differences with their counts and sources, transcribed from the articles.
- `encoders/`: each encoder of the panel with its preprocessing (`image_encoders.py`), and the extraction and caching of the features (`extract_embeddings.py`).
- `controlled/`: the natural and race-balanced pretraining sets (`build_training_mixtures.py`) and the pretraining of the controlled encoders (`train_encoder.py`). `run_one` trains one encoder, and `list_controlled_runs` lists the runs of the controlled design.
- `analysis/`: the disease heads and their thresholds (`heads.py`) and the subgroup measures (`fairness_metrics.py`).
- `mitigation/`: the nine mitigation methods (`preprocessing.py`, `inprocessing.py`, `postprocessing.py`, and `erasure.py`), the Gaussian optimal transport method (`optimal_transport.py`), and the achievable difference at matched disease performance (`ceiling.py`).
- `mechanism/entanglement.py`: linear and nonlinear decodability, the geometric overlap, and the erasure cost.
- `theory/proposition.py`: the synthetic model.
- `experiments/`: one module per analysis. The fair-model reference and its exceedance test for every combination of encoder, finding, and attribute are in `e8_gapnull.py`, and the audit of published subgroup differences is in `e11_published.py`. The mitigation methods and the achievable difference on chest radiographs are in `e2_ceiling.py`, and the same analysis in dermatology and funduscopy is in `e4_generalization.py`. The contrasts between the pretraining conditions of the controlled encoders are in `e1_driver.py`, and the four unfreezing levels are in `e6_finetuning.py`. The association of the representation measures with the achievable difference is in `e3_mechanism.py`, and its transfer to dermatology and funduscopy and the optimal transport method are in `e5_crossmodal.py`. The synthetic model is in `e7_theory.py`, the demographic-signal injection and the disease-signal removal are in `e9_positive_control.py`, and the analyses of acquisition view and site are in `e10_acquisition.py`.
- `merge/build_final_tables.py`: the final tables. `main_build_final_tables` collects the results of every analysis and applies the FDR correction within each family over the results of all analyses together.
- `Inference/`: the bootstrap, the permutation tests, and the FDR correction (`stats_utils.py`), and the result rows, the merging of partial results, and resumption (`report_utils.py`).
- `config/`: the configuration and its reader.

Each module in `experiments/` has one `main_` function that runs the whole analysis. The analyses that are split by encoder or by condition also have a function for one part and a merge function. The merge function applies the FDR correction within each family of tests of its analysis, with `add_fdr` in `Inference/report_utils.py`. The family of the reductions to the achievable difference also holds the tests of the dermatology, funduscopy, and injection analyses. `main_build_final_tables` corrects this family over the pooled tests of the three analyses. Every analysis saves its results per unit of work, skips the units that are already finished, and resumes an interrupted run.

## Data

The chest radiographs come from MIMIC-CXR (with demographics from MIMIC-IV), CheXpert (with demographics from CheXpert Plus), NIH ChestX-ray14, PadChest, VinDr-CXR, and VinDr-PCXR. The dermatology images come from ISIC 2019, Fitzpatrick17k, and the Diverse Dermatology Images set, and the fundus images come from Harvard-FairVision. This repository redistributes none of them. The manifests are not included either, since they contain patient identifiers and demographics from credentialed datasets. Obtain each dataset from its source, then build the manifests with the `main_build_` functions in `data_loader/`. Every dataset and every model comes with a license and terms of use. Following them is the responsibility of the user.

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
