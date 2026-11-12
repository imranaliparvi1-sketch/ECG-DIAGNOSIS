# ECG AI Classifier (phone-friendly web app)

Upload an ECG image, tap Predict, get Normal or History of MI.

## Files
- app.py: the Streamlit app (same crop/resize/normalize as the notebook)
- requirements.txt: Python packages
- ECG_Normal_vs_MI_CNN.keras: YOUR trained model (add this yourself)

## Get a link that opens on your Android phone (free)
1. Download the model from Colab:
   from google.colab import files
   files.download("/content/ECG_CNN_Results/ECG_Normal_vs_MI_CNN.keras")
2. Create a free GitHub account and a new public repository.
3. Upload app.py, requirements.txt and the .keras model to the repository.
4. Go to share.streamlit.io, sign in with GitHub, click "Create app",
   choose your repository, set the main file to app.py, and click Deploy.
5. After a few minutes you get a link like https://your-app.streamlit.app.
   Open it in Chrome on your phone.

## Test on your own computer first
pip install -r requirements.txt
streamlit run app.py

## If the model fails to load
Version mismatch with Colab is the usual cause. Run `print(tf.__version__)` in
Colab and pin that version in requirements.txt (for example tensorflow-cpu==2.19.0).
