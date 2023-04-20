from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load the data
sales_data = pd.read_csv("Grocery.csv")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    product = request.form["product"]

    # Filter data for the selected product and year
    product_data = sales_data[sales_data["Products"] == product]

    # Fill missing values with the most frequent value
    imputer = SimpleImputer(strategy="most_frequent")
    product_data = pd.DataFrame(imputer.fit_transform(
        product_data), columns=product_data.columns)

    # Convert categorical variables into dummy variables
    product_data = pd.get_dummies(
        product_data, columns=["Category", "DayOfWeek"])

    # Split the data into training and testing sets
    train_data = product_data[product_data["Year"]
                              != 2022].drop(columns=["Sales", "Products"])
    train_target = product_data[product_data["Year"] != 2022]["Sales"]
    test_data = product_data[product_data["Year"]
                             == 2022].drop(columns=["Sales", "Products"])
    test_target = product_data[product_data["Year"] == 2022]["Sales"]

    # Train the model
    lr = LinearRegression()
    lr.fit(train_data, train_target)

    # Make predictions
    predictions = lr.predict(test_data)

    return render_template("predict.html", product=product, year=2020, prediction=predictions[0])


if __name__ == "__main__":
    app.run(debug=True)
