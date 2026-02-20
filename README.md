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
تمام! 😎 دلوقتي هجهزلك **ملف README بصيغة Markdown جاهز للـ GitHub** بحيث تقدر تعمل **Copy-Paste مباشر**، وكل التنسيقات هتبان صح:

````markdown
# Car_Evaluation_Project 🚗

## Overview
**Car Evaluation Project** predicts the evaluation of cars based on features such as buying price, maintenance cost, number of doors, seating capacity, luggage boot size, and safety.  
It uses a **Decision Tree Classifier** with both **Gini Index** and **Entropy** criteria, compares their performance, and deploys the best model in a **Streamlit** web application.

---

## Dataset
The dataset contains the following columns:

| Feature   | Description |
|-----------|-------------|
| buying    | Car buying price (`low`, `med`, `high`, `vhigh`) |
| maint     | Maintenance cost (`low`, `med`, `high`, `vhigh`) |
| doors     | Number of doors (`2`, `3`, `4`, `5more`) |
| persons   | Capacity of persons (`2`, `4`, `more`) |
| lug_boot  | Luggage boot size (`small`, `med`, `big`) |
| safety    | Safety rating (`low`, `med`, `high`) |
| class     | Target label (car evaluation) |

---

## Preprocessing
Columns were renamed and encoded using `category_encoders.OrdinalEncoder`.

**Example preprocessing snippet:**
```python
import category_encoders as ce

encoder = ce.OrdinalEncoder(cols=["buying","maint","doors","persons","lug_boot","safety"])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
````

Features were split into `X_train / X_test` and labels into `y_train / y_test`.

---

## Model Training

### 1. Decision Tree Classifier (Gini Index)

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

gini = DecisionTreeClassifier(criterion='gini', max_depth=100, random_state=0)
gini.fit(X_train, y_train)
y_pred_gini = gini.predict(X_test)

print("Testing Accuracy with Gini index:", accuracy_score(y_test, y_pred_gini))
```

* **Testing Accuracy:** 0.9422
* **Training Accuracy:** 1.0000 (slight overfitting observed)

---

### 2. Decision Tree Classifier (Entropy)

```python
entropy = DecisionTreeClassifier(criterion='entropy', max_depth=100, random_state=0)
entropy.fit(X_train, y_train)
y_pred_entropy = entropy.predict(X_test)

print("Testing Accuracy with Entropy:", accuracy_score(y_test, y_pred_entropy))
```

* **Testing Accuracy:** 0.9475 ✅ (better than Gini)

**Confusion Matrix:**

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_entropy)
print(cm)
```

---

## Saving the Model

The best performing model (**Entropy**) and the encoder are saved using `pickle`:

```python
import pickle

with open('model.pkl', 'wb') as file:
    pickle.dump(entropy, file)

with open('encoder.pkl', 'wb') as file:
    pickle.dump(encoder, file)

print("Model saved successfully!")
```

---

## Deployment

The project is deployed as an interactive web app using **Streamlit**:

* Users select values for all features:
  Buying, Maintenance, Doors, Persons, Luggage Boot, Safety
* Click **Predict** to see the predicted car evaluation.
* Requires `model.pkl` and `encoder.pkl` in the project directory.

---

## Requirements

* Python 3.x
* Libraries:

  * `pandas`
  * `scikit-learn`
  * `category_encoders`
  * `streamlit`

Install packages:

```bash
pip install pandas scikit-learn category_encoders streamlit
```

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Car_Evaluation_Project.git
cd Car_Evaluation_Project
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Interact with the app in your browser.

---

## Project Highlights

* Compared **Gini Index** vs **Entropy** for Decision Tree.
* Achieved **94.75% accuracy** with Entropy.
* Saved model and encoder for **Streamlit deployment**.
* Easy-to-use web interface for predictions.

---

## Learning Outcomes 🎯

By completing this project, you will learn to:

### Data Preprocessing

* Encode categorical features using `OrdinalEncoder`.
* Split datasets into training and testing sets.

### Model Training & Selection

* Train Decision Tree models using Gini Index and Entropy.
* Compare models and choose the best-performing one.

### Model Evaluation

* Calculate training and testing accuracy scores.
* Detect overfitting by comparing scores.
* Analyze predictions using a confusion matrix.

### Model Deployment

* Save trained models and encoders using `pickle`.
* Deploy a **Streamlit web app** for interactive predictions.

### End-to-End Machine Learning Pipeline

* Gain hands-on experience with a complete workflow from **preprocessing → training → evaluation → deployment**.

```


