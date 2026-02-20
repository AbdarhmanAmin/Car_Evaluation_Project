# Car Evaluation Project 🚗

This project focuses on predicting the quality/evaluation of cars based on several technical and price-related attributes. I implemented a **Decision Tree Classifier** using different splitting criteria (Gini Index and Entropy) to determine the most accurate model for classification.

## 📋 Project Overview
The goal of this project is to classify cars into different evaluation categories using a dataset that includes features like buying price, maintenance cost, number of doors, safety, and more.

## 🗂️ Dataset Features
The dataset consists of the following features:
* `buying`: Buying price.
* `maint`: Price of the maintenance.
* `doors`: Number of doors.
* `persons`: Capacity in terms of persons to carry.
* `lug_boot`: The size of luggage boot.
* `safety`: Estimated safety of the car.
* **Target (`class`)**: The evaluation of the car.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** `pandas`, `scikit-learn`, `category_encoders`, `pickle`.

---

## 🚀 Workflow

### 1. Data Preprocessing
Since the features are categorical, I used **Ordinal Encoding** to transform them into numerical values that the machine learning model can understand.
```python
import category_encoders as ce
encoder = ce.OrdinalEncoder(cols=["buying","maint","doors","persons","lug_boot","safety"])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
2. Modeling & Comparison
I implemented two versions of the Decision Tree Classifier to find the optimal result:

A. Decision Tree with Gini Index
Accuracy Score: 94.22%

Training Set Score: 100.00%

Observation: The model showed a slight sign of overfitting as it perfectly fit the training data.

B. Decision Tree with Entropy (Selected Model)
Accuracy Score: 94.75%

Training Set Score: 100.00%

Observation: This model performed slightly better on the test set and provided a more balanced result compared to the Gini Index model.

3. Model Evaluation
I generated a Confusion Matrix for the Entropy-based model to visualize the performance across different classes and ensure the model's reliability in predicting each category.

💾 Saving the Model
Since the Entropy model yielded the highest accuracy, I selected it for final use. I saved both the model and the encoder using pickle for easy deployment and future predictions:

model.pkl: The trained Decision Tree (Entropy).

encoder.pkl: The fitted Ordinal Encoder.

Python
import pickle
# Saving the model and encoder
with open('model.pkl', 'wb') as file:
    pickle.dump(entropy, file)

with open('encoder.pkl', 'wb') as file:
    pickle.dump(encoder, file)
🎓 Learning Outcomes
Through this project, I have gained hands-on experience in:

Feature Engineering: Applying OrdinalEncoder to handle categorical data effectively.

Hyperparameter Analysis: Comparing Gini Index vs. Entropy to select the best splitting criterion for a Decision Tree.

Model Evaluation: Utilizing accuracy_score and confusion_matrix to diagnose model performance.

Overfitting Diagnostics: Analyzing the gap between training and testing accuracy.

Model Serialization: Exporting models and encoders using pickle for production readiness.

🏁 Conclusion
The Decision Tree Classifier with the Entropy criterion proved to be the most effective for this dataset, achieving an accuracy of ~94.75%. This project serves as a complete end-to-end Machine Learning pipeline developed independently.

Developed by [Your Name]
