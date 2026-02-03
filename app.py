from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

model = joblib.load("linear_svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
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
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

