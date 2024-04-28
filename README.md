# Bank Customer Churn Prediction using Artificial Neural Networks


Introduction
This repository contains code and resources for predicting customer churn in a bank using Artificial Neural Networks (ANN). Customer churn, the rate at which customers stop doing business with a company, is a critical metric for businesses to understand and mitigate. In this project, we aim to build a predictive model that identifies customers who are more likely to leave the bank soon, based on historical data.

Dataset
The dataset provided contains 10,000 records of bank customers, including various features such as customer ID, credit score, geography, gender, age, tenure, balance, number of products, whether the customer has a credit card, is an active member, and their estimated salary. The target variable is whether the customer exited the bank (1 for exited, 0 for retained).

Requirements
Python 3.x
NumPy
Pandas
TensorFlow
Keras
Scikit-learn
Setup
Clone this repository:
bash
Copy code
git clone https://github.com/your_username/bank-customer-churn-prediction.git
Install the required dependencies:
Copy code
pip install -r requirements.txt
Download the dataset (bank_dataset.csv) and place it in the data/ directory.
Usage
Run the train.py script to train the ANN model:
Copy code
python train.py
Once the model is trained, you can use the predict.py script to make predictions on new data:
Copy code
python predict.py
Model Architecture
The ANN model architecture consists of multiple layers of densely connected neurons, with dropout layers to prevent overfitting. The input layer takes the feature vectors, followed by several hidden layers with ReLU activation functions. The output layer uses a sigmoid activation function to produce a binary classification output indicating whether the customer is likely to churn or not.

Evaluation
The model performance can be evaluated using metrics such as accuracy, precision, recall, and F1-score. Additionally, the receiver operating characteristic (ROC) curve and the area under the curve (AUC) can provide insights into the model's ability to discriminate between classes.

Conclusion
Predicting customer churn is crucial for businesses to retain their customers and maintain profitability. By leveraging Artificial Neural Networks, we can build predictive models that help identify customers at risk of leaving the bank. This project serves as a starting point for implementing such predictive systems in real-world scenarios.
