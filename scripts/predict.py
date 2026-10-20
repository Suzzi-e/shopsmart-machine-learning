
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Locating the saved models
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Loading trained models
regression_model = joblib.load(
    MODELS_DIR / "linear_regression.pkl"
)

classification_model = joblib.load(
    MODELS_DIR / "logistic_regression.pkl"
)

# Loading fitted preprocessing objects
encoders = joblib.load(MODELS_DIR / "encoder.pkl")
scalers = joblib.load(MODELS_DIR / "scaler.pkl")

regression_encoder = encoders["regression_encoder"]
classification_encoder = encoders["classification_encoder"]

regression_scaler = scalers["regression_scaler"]
classification_scaler = scalers["classification_scaler"]


def get_number(prompt, minimum=None):
    """Ask for a numeric input and validate it."""
    while True:
        try:
            value = float(input(prompt))

            if not np.isfinite(value):
                print("Please enter a valid finite number.")
                continue

            if minimum is not None and value < minimum:
                print(f"Please enter a value of at least {minimum}.")
                continue

            return value

        except ValueError:
            print("Please enter a valid number.")


def get_choice(prompt, choices):
    """Ask the user to select a category."""
    print(f"\n{prompt}")
    for i, choice in enumerate(choices, start=1):
        print(f"{i}. {choice}")

    while True:
        selection = input("Enter the number of your choice: ").strip()

        if selection.isdigit():
            index = int(selection) - 1
            if 0 <= index < len(choices):
                return choices[index]

        print("Invalid choice. Please select one of the listed numbers.")


def main():
    print("\n=== ShopSmart Order Prediction ===")
    print("Enter the details for the order.\n")

    # Get user input
    quantity = get_number("Quantity: ", minimum=0)
    unit_price = get_number("Unit price: ", minimum=0)
    discount = get_number("Discount amount: ", minimum=0)
    shipping_cost = get_number("Shipping cost: ", minimum=0)

    customer_segment = get_choice(
        "Customer segment",
        ["Bronze", "Silver", "Gold"]
    )

    payment_method = get_choice(
        "Payment method",
        ["Cash", "Card", "Transfer", "Mobile Money"]
    )

    sales_channel = get_choice(
        "Sales channel",
        ["Online", "In-Store"]
    )

    category = get_choice(
        "Product category",
        ["Electronics", "Home", "Groceries", "Clothing"]
    )

    region = get_choice(
        "Region",
        ["North", "South", "East", "West"]
    )

    # Preparing categorical inputs
    categorical_data = pd.DataFrame([{
        "Customer_Segment": customer_segment,
        "Payment_Method": payment_method,
        "Sales_Channel": sales_channel,
        "Category": category,
        "Region": region
    }])

    # Preparing numerical inputs for regression
    quantity_price = quantity * unit_price

    regression_numeric_data = pd.DataFrame([{
        "Quantity_order": quantity,
        "Unit_Price_order": unit_price,
        "Discount_order": discount,
        "Shipping_Cost": shipping_cost,
        "Quantity_Price": quantity_price
    }])

    # Preparing numerical inputs for classification
    classification_numeric_data = pd.DataFrame([{
        "Quantity_order": quantity,
        "Unit_Price_order": unit_price,
        "Discount_order": discount,
        "Shipping_Cost": shipping_cost
    }])

    # Encoding and scale using the fitted objects
    regression_categorical = regression_encoder.transform(
        categorical_data
    )
    regression_numeric = regression_scaler.transform(
        regression_numeric_data
    )

    classification_categorical = classification_encoder.transform(
        categorical_data
    )
    classification_numeric = classification_scaler.transform(
        classification_numeric_data
    )

    # Combining features in the same order used during training
    regression_input = np.hstack([
        regression_categorical,
        regression_numeric
    ])

    classification_input = np.hstack([
        classification_categorical,
        classification_numeric
    ])

    # Making predictions
    predicted_order_value = regression_model.predict(
        regression_input
    )[0]

    probabilities = classification_model.predict_proba(
        classification_input
    )[0]

    # Finding the probability belonging to class 1 (returned)
    returned_class_index = list(
        classification_model.classes_
    ).index(1)

    return_probability = probabilities[returned_class_index]

    # Displaying the results
    print("\n=== Prediction Results ===")
    print(f"Predicted order value: {predicted_order_value:,.2f}")
    print(f"Return probability: {return_probability:.1%}")

    print("\nNote: These are model estimates, not guarantees.")
    print("\n")

if __name__ == "__main__":
    main()