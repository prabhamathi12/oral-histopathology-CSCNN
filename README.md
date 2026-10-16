# Untangling the stains: decorrelation contrast stretching and a lightweight CNN for oral histopathology

Code, results and figures for the manuscript

> Prabhakaran M and coauthors *Untangling the stains: decorrelation contrast stretching and a lightweight CNN for oral histopathology.* Submitted to *Scientific Reports*.

`cscnn_pipeline.py` reproduces every result in the paper:

- **Enhancement study** – decorrelation contrast stretching vs linear stretching, histogram equalisation, CLAHE, Reinhard and Macenko normalisation on seven image-quality measures.
- **Augmentation-order experiment** – the same network trained with augmentation before and after the train–test split.
- **CSCNN** – a five-block network (70,644 parameters) with a fixed decorrelation-stretch input layer, five-fold cross-validation and three ablations.
- **Baselines** – 16 ImageNet-pretrained networks on the same test images, with McNemar tests, bootstrap confidence intervals, Grad-CAM and t-SNE.

## Contents

| Path | Description |
|---|---|
| `cscnn_pipeline.py` | Complete pipeline: data preparation, experiments, figures and tables |
| `make_paper_figures.py` | Redraws the summary figures from the pipeline outputs |
| `notebooks/run_in_colab.ipynb` | Ready-to-run Google Colab notebook |
| `results/` | Summary tables for each dataset and the metrics of every CSCNN run |
| `figures/` | Figures 1–18 of the paper |

## Data

| Dataset | Used in this study | Source |
|---|---|---|
| Tongue biopsies | 712 H&E images, 4 classes (tongue subset of a 12-class oral and nasal cavity collection) | Prabhakaran M, Mendeley Data V2 (2026), [doi:10.17632/6kz483gnk3.2](https://doi.org/10.17632/6kz483gnk3.2) |
| NDB-UFES | 237 images, 69 patients: OSCC, leukoplakia with / without dysplasia | Ribeiro-de-Assis *et al.*, *Data in Brief* 48, 109128 (2023) |
| Rahman *et al.* | 1,196 images: normal oral epithelium and OSCC, 100× and 400× | Rahman *et al.*, *Data in Brief* 29, 105114 (2020) |

Download the datasets and place them as follows (one folder per class):

```
TONGUE-RAW/<class>/<images>
NDB-UFES/<release folder>/ndb-ufes.csv, images.zip
ftmp4cvtmb-1/First Set/, Second Set/
```

## Quick start

**Google Colab** (GPU runtime): open `notebooks/run_in_colab.ipynb` and run the cells in order.

**Local machine** (Python 3.10+, CUDA GPU recommended):

```bash
git clone https://github.com/prabhamathi12/oral-histopathology-cscnn.git
cd oral-histopathology-cscnn
pip install -r requirements.txt
python cscnn_pipeline.py
```

Before running, set `WORK_ROOT` (output folder) and `DATASET_DIRS` (data folders) at the top of `cscnn_pipeline.py`. Set `QUICK_TEST = True` to check the setup with a two-epoch run. Completed runs are skipped, so an interrupted job resumes where it stopped.

## Main settings

| Setting | Default | Meaning |
|---|---|---|
| `DATASETS_TO_RUN` | `["tongue", "ndbufes", "rahman"]` | Datasets processed in turn |
| `STEPS` | all six | `enhancement`, `leakage_demo`, `cscnn`, `ablations`, `baselines`, `figures` |
| `CV_FOLDS` | `5` | Stratified, group-aware cross-validation |
| `BASELINE_FOLDS` | `1` | Folds used for the 16 pretrained networks |
| `BATCH`, `IMG_SIZE`, `EPOCHS` | `16`, `224`, `100` | Training settings shared by all models |

## Evaluation protocol

- Byte-identical images are grouped by MD5 checksum and always kept in the same fold.
- Data are divided before any augmentation: by patient for NDB-UFES and by image group for the other datasets. An assertion stops the run if a group appears in two subsets.
- Augmentation is applied on the fly to training images only.
- Models are selected on validation balanced accuracy; each test fold is used once.

## Results and paper items

| Paper | File in `results/<dataset>/` |
|---|---|
| Table 6, Figs 5–6, Supplementary Table S1 | `Table_enh_summary.csv`, `Table_enh_friedman.csv`, `Table_enh_wilcoxon.csv`, `Table_enh_mean_ranks.csv` |
| Table 7, Fig. 7 | `Table_leakage_demo.csv`, `leakage_demo_all_runs.csv` (tongue) |
| Tables 8 and 10, Supplementary Table S2 | `cscnn_runs/<model>/seed1_fold<k>_metrics.json` |
| Table 9, Figs 10–16 | `Table_model_comparison_fold1_all_models.csv` |
| Supplementary Tables S3–S5 | `Table_per_class.csv`, `Table_mcnemar.csv`, `Table_per_magnification.csv` (Rahman) |

## Citation

Please cite the paper and the tongue biopsy dataset if you use this code (see `CITATION.cff`).

## Licence

Code is released under the MIT Licence. The datasets are subject to the terms of their original publications.
