# 📱 Mobile Price Range Predictor

![Model Status](https://img.shields.io/badge/Model-RandomForest-blue) ![Python](https://img.shields.io/badge/Made%20with-Python%203.12-yellow) ![License](https://img.shields.io/badge/License-MIT-green)

A machine learning model that predicts whether a smartphone falls into a **low**, **medium**, **high**, or **very high** price category based on its specifications. Built using Random Forest Classifier in Python, this project showcases practical use of classification algorithms for market segmentation.

---

## 🚀 Project Highlights

- ✅ Uses 2000 real-world mobile phone entries
- 🔎 Analyzes 20+ features like RAM, 4G, battery, screen size, camera, etc.
- 📊 Includes EDA & feature importance analysis
- 🤖 Trained using RandomForestClassifier (Accuracy ~89.25%)
- 📈 Evaluation: Confusion matrix & classification report

---

## 📁 Dataset Overview

The dataset includes the following features:
- `battery_power`, `blue`, `clock_speed`, `dual_sim`, `fc`, `four_g`, `int_memory`, `m_dep`
- `mobile_wt`, `n_cores`, `pc`, `px_height`, `px_width`, `ram`, `sc_h`, `sc_w`, `talk_time`
- `three_g`, `touch_screen`, `wifi`
- 🔚 `price_range`: 0 = low, 1 = medium, 2 = high, 3 = very high (target)

---

## 🛠️ Tech Stack

- Python 3.12
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn (Random Forest)

---

## 🔍 Sample Insights

```bash
Accuracy: 0.8925

Confusion Matrix:
 [[101   4   0   0]
  [  5  79   7   0]
  [  0   6  80   6]
  [  0   0  15  97]]
```

- Class 0 (Low): 95% precision
- Class 3 (Very High): 94% precision

---

## 📊 Feature Importance (Top Influencers)
- `ram`
- `battery_power`
- `px_height`
- `px_width`
- `mobile_wt`

---

## ✅ How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/mobile-price-range-predictor.git
   cd mobile-price-range-predictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the model:
   ```bash
   python predict_mob_pricing.py
   ```

---

## 📈 Future Enhancements

- Add a Flask or Streamlit web app for UI-based predictions
- Integrate hyperparameter tuning (GridSearchCV)
- Add XGBoost & deep learning comparisons

---

## 📜 License

Licensed under the [MIT License](LICENSE).

---

If you find this project useful, please ⭐ the repo or fork it to build your own version!
