from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)


vectorizer = None
model = None

def load_model():
    global vectorizer, model
    if vectorizer is None or model is None:
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        model = joblib.load("linear_svm_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    load_model()
    data = request.get_json()
    text = data["review"]

    X = vectorizer.transform([text])
    pred = model.predict(X)[0]

    sentiment = "positive" if pred == 1 else "negative"

    return jsonify({
        "sentiment": sentiment,
        "prediction": int(pred)
    })

'''if __name__ == "__main__":
    app.run(debug=True)'''
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


