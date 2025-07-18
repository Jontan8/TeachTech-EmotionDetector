#1) Mount the Google Drive and verify the path installation:
from google.colab import drive
import os
import pandas as pd

drive.mount('/content/drive', force_remount=True)
CSV_PATH = '/content/drive/MyDrive/emotions.csv'
assert os.path.exists(CSV_PATH), f"File not found:" ;{CSV_PATH}

#2) Inspect the CSV Header to confirm the column names.
df_header = pd.read_csv(CSV_PATH, nrows = 0)
print("CSV Columns:", df_header.columns.tolist)

#3) Import Dependancies
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib

#4) Configure (Updating to match CSV file)
CHUNKSIZE = 1000 #setting rows per batch
TEXT_COL = 'text' # name of the text column in your CSV
LABEL_COL = 'label' #name of the label column in your CSV
N_FEATURES = 2**18 #number of hashed features in your model
MODEL_PATH = '/content/drive/MyDrive/emotion_clf.pkl'

#5) Training Model
def train_and_save(csv_path, model_path):
  #verifies these paths exist
  header = pd.read_csv(csv_path, nrows = 0).columns.tolist()
  if TEXT_COL not in header or LABEL_COL not in header:
    raise KeyError (f"Required columns not found. Available: {header}")
    vectorizer = HashingVectorizer(
        n_features = N_FEATURES,
        alternate_sign = False,
        norm = None,
        binary = False

    )
    encoder = LabelEncoder() # Instantiate LabelEncoder
    classifier = SGDClassifier(
        loss = 'log_loss', #logistic regression
        max_iter = 1,
        tol = None,
        learning_rate = 'optimal',
        random_state = 42
    )
    first_pass = True
    classes = None
    for chunk in pd.read_csv(csv_path, chunksize = CHUNKSIZE):
      texts = chunk[TEXT_COL].astype(str).tolist()
      X = vectorizer.transform(texts)
      y_raw = chunk[LABEL_COL].values

      if first_pass:
        encoder.fit(y_raw)
        classes = encoder.transform(encoder.classes_)
        first_pass = False

      y = encoder.transform(y_raw)
      if not hasattr(classifier, "classes_"):
        classifier.partial_fit(X, y, classes=classes) # Pass classes for the first fit
      else:
        classifier.partial_fit(X, y)


    joblib.dump({'model': classifier, 'vectorizer': vectorizer, 'encoder': encoder}, model_path)
    print(f"Traning Finished. Model has been saved to {model_path}")

# Prediction Function:
def load_and_predict(text,model_path):/filepath to java
  data = joblib.load(model_path)
  model = data['model']
  vectorizer = data['vectorizer'] # Corrected key
  encoder = data['encoder'] # Corrected key

  X_new = vectorizer.transform([text])
  y_pred = model.predict(X_new)
  return encoder.inverse_transform(y_pred)[0]

#7) Execute training and predict
train_and_save(CSV_PATH, MODEL_PATH) # Corrected function call
sample_text = "I am happy"
predicted_emotion = load_and_predict(sample_text, MODEL_PATH)
print(f"Input: {sample_text}\nPredicted Emotion: {predicted_emotion}")
