from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Load a small sample of the training data once at startup, purely so the
# result page can plot a new prediction alongside real data points
# (income vs. price). This doesn't affect the prediction itself.
housing = pd.read_csv("housing.csv")
sample = housing[["Avg. Area Income", "Price"]].sample(n=60, random_state=7)
SAMPLE_POINTS = [
    {"income": round(row["Avg. Area Income"]), "price": round(row["Price"])}
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