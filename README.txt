HEALTHAI STREAMLIT DEPLOYMENT

Files:
- app.py
- requirements.txt
- .python-version

Put these files in the root of the GitHub repository, beside:
- HWRPA_Model.pkl
- HWRPA_ROC_Curve.png
- HWRPA_Confusion_Matrix.png
- Feature_Importance.png
- HWRPA_Performance_Graph.png
- your model comparison chart (optional)
- CSV result files (optional)

IMPORTANT:
1. The app expects HWRPA_Model.pkl to be the dictionary saved by the notebook with keys:
   models, weights, features, threshold (and optionally target).
2. The app does not upload or read the original CDC dataset.
3. Use Python 3.11 for the Streamlit app runtime. The saved scikit-learn model was created
   with scikit-learn 1.6.1; matching versions is important for pickle/joblib compatibility.
4. Restart/reboot the Streamlit app after committing all files.
5. Only load pickle/joblib files that you created and trust.

Run locally:
  pip install -r requirements.txt
  streamlit run app.py

The dashboard is a research prototype, not a diagnostic tool or future-disease predictor.
