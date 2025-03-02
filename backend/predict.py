import joblib
import sys
import json
import numpy as np

# Debug: Print the command-line arguments
print("Command-line arguments:", sys.argv)

# Load the trained model
model_path = "../ml-model/models/mental_health_model.pkl"
print("Loading model from:", model_path)
model = joblib.load(model_path)

# Read input data from command line arguments
try:
    # Combine all arguments after the script name into a single string
    input_json = " ".join(sys.argv[1:])
    print("Input JSON string:", input_json)

    # Remove any extra quotes or spaces
    input_json = input_json.strip().strip("'").strip('"')
    print("Cleaned input JSON string:", input_json)

    # Parse the JSON input
    input_data = json.loads(input_json)
    print("Parsed input data:", input_data)

    # Ensure all 20 features are present
    features = [
        input_data.get("Age", 0),
        input_data.get("Number of Children", 0),
        input_data.get("Physical Activity Level", 0),
        input_data.get("Employment Status", 0),
        input_data.get("Income", 0),
        input_data.get("Alcohol Consumption", 0),
        input_data.get("Dietary Habits", 0),
        input_data.get("Sleep Patterns", 0),
        input_data.get("History of Mental Illness", 0),
        input_data.get("History of Substance Abuse", 0),
        input_data.get("Family History of Depression", 0),
        input_data.get("Chronic Medical Conditions", 0),
        input_data.get("Marital Status", 0),
        input_data.get("Educational Level", 0),
        input_data.get("Smoking Status", 0),
        0,  # Add missing features
        0,
        0,
        0,
        0
    ]

    # Convert input data to a NumPy array (required by the model)
    input_array = np.array([features], dtype=np.float32)
    print("Input array for prediction:", input_array)
except json.JSONDecodeError as e:
    print("Error decoding JSON:", e)
    sys.exit(1)

# Make prediction
try:
    prediction = model.predict(input_array)
    print("Prediction:", prediction)
except Exception as e:
    print("Error during prediction:", e)
    sys.exit(1)

# Return the prediction as JSON
# Convert the prediction to a standard Python data type
prediction_value = prediction[0].tolist()  # Convert NumPy array to list
print(json.dumps({"prediction": prediction_value}))