# MED-01 — Hospital Appointment No-Show Prediction

Predicts the likelihood that a scheduled medical appointment will result in a no-show, 
to help clinic staff prioritize reminders and follow-up.

## Contents
- `MED01_NoShow_Prediction.ipynb` — full pipeline: data cleaning, exploration, modeling, evaluation, inference function
- `noshow_appointments.csv` — dataset (Kaggle "Medical Appointment No Shows", Brazil 2016)
- `noshow_model.pkl` — trained Random Forest model
- `MED-01-completed.docx` — project brief with Data & Problem Discovery and Technical Proposal

## How to run
Open the notebook in Google Colab, upload `noshow_appointments.csv` when prompted, and run all cells in order.

## Results
- Baseline (Logistic Regression): ROC-AUC 0.659
- Final model (Random Forest): ROC-AUC 0.710
- Key finding: scheduling lead time (WaitDays) is the dominant predictor of no-shows

## Limitations
Trained on 2016 Brazilian public-clinic data; precision on the no-show class is moderate (~0.31), 
so predictions should support reminder triage, not automatic decisions.
