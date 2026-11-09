# 🦟 Malaria Detection from Blood Smear Images (CNN)

A custom Convolutional Neural Network that classifies single-cell blood smear images as **Parasitized** or **Uninfected**, reaching **95.79% test accuracy** and **0.99 AUC** on a held-out, perfectly balanced test set of 4,136 images. The trained model is served through an interactive **Streamlit** web app.

## 📊 Results (held-out test set, 4,136 images)

| Metric | Value |
|---|---|
| Test accuracy | **95.79%** |
| Test AUC | **0.9908** |
| Test loss | 0.1159 |
| Precision / Recall (Parasitized) | 0.97 / 0.94 |
| Precision / Recall (Uninfected) | 0.94 / 0.97 |
| F1-score (both classes) | 0.96 |

**Confusion matrix** (rows = actual, columns = predicted)

| | Pred. Parasitized | Pred. Uninfected |
|---|---|---|
| **Actual Parasitized** | 1949 | 119 |
| **Actual Uninfected** | 55 | 2013 |

Only 174 of 4,136 test images were misclassified. In a medical setting, the 119 missed infections (false negatives) matter more than the 55 false alarms, so lowering the decision threshold to raise recall is the natural next step (see Future Work).

## 🗂️ Dataset

- Cell Images for Detecting Malaria (NIH / Kaggle): **27,558 images**, 13,779 per class (perfectly balanced).
- Split with a fixed seed (42): **70% train (19,290) / 15% validation (4,132) / 15% test (4,136)**.
- Images resized to **130 × 130 × 3** and rescaled to [0, 1]. Batch size: 32.

## 🧠 Pipeline

1. **Data validation:** OpenCV was used to verify every image can be opened (invalid extensions, zero-byte and unreadable files are detected and removed; none were found).
2. **EDA:** class balance check and random image samples per class.
3. **Baseline CNN:** 3 Conv blocks (32/64/128 filters) + Flatten + Dense(128).
4. **Overfitting diagnosis:** the baseline reached ~99.5% training accuracy but validation accuracy peaked around 95.3% and then dropped to 93.4% (validation loss rose from 0.15 to 0.33), a clear overfitting signal.
5. **Optimized CNN:**
   - 7×7 kernel in the first Conv layer, `he_normal` initialization
   - `GlobalAveragePooling2D` instead of `Flatten` (far fewer parameters)
   - Dense(64) + **Dropout(0.3)**
   - **EarlyStopping** (patience 5 on validation loss, best weights restored from epoch 18) and **ModelCheckpoint**
   - AUC tracked alongside accuracy
6. **Evaluation:** classification report, confusion matrix, and visual inspection of misclassified images.
7. **Deployment:** Streamlit web app for uploading a cell image and getting a prediction.

## 🛠️ Tech Stack

TensorFlow / Keras, OpenCV, NumPy, Pandas, Matplotlib, Seaborn, split-folders, Streamlit

## 📁 Repository Structure

```
CNN_Malaria/
├── Malaria_CNN_Classifier.ipynb   # Full pipeline: EDA, training, evaluation
├── app.py                         # Streamlit web app
├── malaria_cnn_model.keras        # Final trained model
└── requirements.txt
```

## 🚀 Quickstart

```bash
git clone https://github.com/habiba-10/CNN_Malaria.git
cd CNN_Malaria
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

To retrain, download the dataset into a `cell_images/` folder (with `Parasitized/` and `Uninfected/` subfolders) and run the notebook.

## ⚠️ Limitations

- Trained and tested on one dataset of segmented single-cell images; performance on images from other microscopes, stains, or labs is untested.
- This is an educational project and **not a certified diagnostic tool**.

## 🔭 Future Work

- Tune the decision threshold to prioritize recall for the Parasitized class.
- Add data augmentation and compare against transfer learning (e.g., MobileNet / EfficientNet).
- Add Grad-CAM visualizations to show which cell regions drive each prediction.

## 👤 Author

**Habiba Ezat** · [GitHub](https://github.com/habiba-10) · [LinkedIn](https://www.linkedin.com/in/habiba-ezat-a39521293)
