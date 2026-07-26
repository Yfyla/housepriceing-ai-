from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model
with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

# Load dataset
housing = pd.read_csv(os.path.join(BASE_DIR, "housing.csv"))

# Create sample points for graph
sample = housing[["Avg. Area Income", "Price"]].sample(n=60, random_state=7)

SAMPLE_POINTS = [
    {
        "income": round(row["Avg. Area Income"]),
        "price": round(row["Price"])
    }
    for _, row in sample.iterrows()
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    income = float(request.form["income"])
    age = float(request.form["age"])
    rooms = float(request.form["rooms"])
    bedrooms = float(request.form["bedrooms"])
    population = float(request.form["population"])

    prediction = model.predict([[
        income,
        age,
        rooms,
        bedrooms,
        population
    ]])

    price = round(prediction[0])

    return render_template(
        "result.html",
        price=price,
        income=income,
        age=age,
        rooms=rooms,
        bedrooms=bedrooms,
        population=population,
        sample_points=SAMPLE_POINTS
    )


if __name__ == "__main__":
    app.run(debug=True)