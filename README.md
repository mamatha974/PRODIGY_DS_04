# PRODIGY_DS_04
# 🐦 Twitter Sentiment Analysis

This project uses machine learning to classify tweets into **Positive**, **Negative**, or **Neutral** sentiments. It uses two datasets: one for training and one for validation. The model is trained using Logistic Regression and evaluated using standard metrics.

---

## 📁 Files

- `twitter_training.csv` – Training dataset  
- `twitter_validation.csv` – Validation dataset  
- `twitter.csv` – Trained model  
- `tfidf_vectorizer.pkl` – TF-IDF vectorizer  
- `README.md` – Project overview (this file)

---

## 🔧 Technologies Used

- Python, Google Colab  
- Pandas, Scikit-learn  
- TF-IDF Vectorizer  
- Logistic Regression  
- Joblib (for saving model)

---

## ⚙️ How It Works

1. **Upload datasets** to Colab  
2. **Clean tweets** (remove links, symbols, etc.)  
3. **Convert text to numbers** using TF-IDF  
4. **Train model** using Logistic Regression  
5. **Evaluate model** using precision, recall, f1-score  
6. **Save model** for future predictions

---

## 🧪 Sample Prediction Code

```python
import joblib
model = joblib.load('twitter_sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

tweet = ["I love this product!"]
cleaned = ["i love this product"]
vec = vectorizer.transform(cleaned)
pred = model.predict(vec)
**Mamatha Jonnadula**
Project made as part of Data Science learning.

yaml
Copy
Edit

---

You can now copy this and paste it directly into a `README.md` file for GitHub or any project folder.

Let me know if you want:
- A GitHub upload guide
- A cover image/banner for the README
- To turn this into a simple Streamlit web app
print("Predicted Sentiment:", pred)

