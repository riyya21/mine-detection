{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "e84e87bc-db33-4851-8f43-806af7d930a4",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import numpy as np\n",
    "import joblib\n",
    "\n",
    "# Load the pre-trained model and scaler\n",
    "model = joblib.load(\"logistic_regression_model.pkl\")\n",
    "scaler = joblib.load(\"scaler.pkl\")\n",
    "\n",
    "# Streamlit app title\n",
    "st.title(\"Rock vs Mine Prediction App\")\n",
    "st.write(\"This app predicts whether the object is a **Rock** or a **Mine** based on sonar features.\")\n",
    "\n",
    "# Input data\n",
    "st.header(\"Input Features\")\n",
    "st.write(\"Enter 60 sonar features (comma-separated):\")\n",
    "input_data = st.text_area(\"Features Input\", placeholder=\"e.g., 0.02, 0.03, 0.45, ..., 0.67\")\n",
    "\n",
    "# Prediction button\n",
    "if st.button(\"Predict\"):\n",
    "    try:\n",
    "        # Convert input into a numpy array\n",
    "        features = np.array(input_data.split(','), dtype=float).reshape(1, -1)\n",
    "        \n",
    "        # Scale the input data\n",
    "        features_scaled = scaler.transform(features)\n",
    "        \n",
    "        # Make a prediction\n",
    "        prediction = model.predict(features_scaled)\n",
    "        \n",
    "        # Display the result\n",
    "        if prediction[0] == 'R':\n",
    "            st.success(\"Prediction: **Rock**\")\n",
    "        else:\n",
    "            st.success(\"Prediction: **Mine**\")\n",
    "    except ValueError:\n",
    "        st.error(\"Invalid input. Please ensure you enter 60 comma-separated numerical values.\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
