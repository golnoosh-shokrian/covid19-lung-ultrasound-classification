# COVID-19 Detection from Lung Ultrasound Videos with Deep Learning
MSc thesis project (2023) — Hamedan University of Technology, Biomedical Engineering (Bioelectric).
Classification of COVID-19 from lung ultrasound videos by converting videos to frames and fine-tuning CNNs in Python/TensorFlow.

## Method
- Video-to-frame conversion from lung ultrasound (linear and convex probes)
- Preprocessing and dataset cleaning in Python, analysis in MATLAB
- Compared transfer-learning models: InceptionV3, Xception, ResNet50V2
- Best result with fine-tuned InceptionV3
- Evaluation with cross-validation + Grad-CAM for interpretability

## Key files
- `build_image_dataset.py` — build frame dataset from videos
- `train_covid19.py` — main training script
- `train_comet_covid19.py` — training variant with experiment tracking
- `evaluate_iclus.py` / `eval_vid_classifier.py` — evaluation
- `cross_val_splitter.py` — cross-validation splits
- `generate_gradcams.py` — Grad-CAM visualization
- `dataset_cleanup.py` / `add_uninformative_class.py` — data cleaning
- `make_plots_from_results.py` / `iclus_results.py` / `make_latex_tables.py` — plots and tables

## Results

| Model | Accuracy | Sensitivity | Specificity |
| InceptionV3 (fine-tuned) | 0.76 | 
| Xception |0.72 | 
| ResNet50V2 | 0.71 |

## How to run
```bash
pip install -r requirements.txt
python build_image_dataset.py
python train_covid19.py
python evaluate_iclus.py

python build_image_dataset.py
python train_covid19.py
python evaluate_iclus.py
