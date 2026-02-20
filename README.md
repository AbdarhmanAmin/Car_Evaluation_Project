# Car Evaluation Project 🚗

In this project, I built a predictive model to evaluate car quality based on specific technical and financial features. I explored the **Decision Tree Classifier** using different splitting criteria to achieve the highest possible accuracy.

## 📊 Project Workflow

### 1. Data Preprocessing & Encoding

Since the dataset contains categorical features, I used **Ordinal Encoding** to convert them into a numerical format suitable for the model while preserving the inherent order of the categories.

```python
import category_encoders as ce

# Initializing and applying the Ordinal Encoder
encoder = ce.OrdinalEncoder(cols=["buying", "maint", "doors", "persons", "lug_boot", "safety"])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

```

---

### 2. Modeling & Comparison

I experimented with two different mathematical criteria for the Decision Tree to see which one performs better on this specific dataset:

#### A. Decision Tree with Gini Index

The Gini index measures the probability of a specific variable being wrongly classified when chosen randomly.

* **Accuracy Score:** 94.22%
* **Training Set Score:** 100.00%
* **Observation:** The model showed a slight sign of overfitting since it achieved perfect accuracy on training data but slightly less on testing.

#### B. Decision Tree with Entropy (Selected Model)

Entropy measures the impurity or randomness in the data.

* **Accuracy Score:** 94.75%
* **Training Set Score:** 100.00%
* **Observation:** This model outperformed the Gini index approach, providing better generalization on the test set.

---

### 3. Mathematical Foundations

For this project, I compared two main criteria:

* **Gini Index:** 
* **Entropy:** 

---

### 4. Model Evaluation

To ensure the model isn't just "guessing," I generated a **Confusion Matrix** for the Entropy-based model. This allowed me to see exactly where the model was making correct predictions and where it was tripping up across different classes.

---

## 💾 Saving the Model

After concluding that the **Entropy model** was the most effective, I exported both the model and the encoder using `pickle`. This ensures that the preprocessing steps are consistent when the model is deployed.

```python
import pickle

# Saving the trained entropy model
with open('model.pkl', 'wb') as file:
    pickle.dump(entropy, file)

# Saving the fitted encoder
with open('encoder.pkl', 'wb') as file:
    pickle.dump(encoder, file)

print("The model and encoder have been saved successfully!")

```

---

## 🎓 Learning Outcomes

Through this project, I have successfully demonstrated:

* **Feature Engineering:** Mastering `OrdinalEncoder` for handling non-numeric data.
* **Algorithm Tuning:** Comparing Gini vs. Entropy to optimize Decision Tree performance.
* **Performance Metrics:** Using `accuracy_score` and `confusion_matrix` for deep model analysis.
* **Overfitting Management:** Identifying and analyzing the gap between training and testing performance.
* **Model Deployment Readiness:** Using `pickle` for model serialization.

---

## 🏁 Conclusion

The Decision Tree Classifier using the **Entropy** criterion achieved a robust accuracy of **94.75%**. This project highlights a complete end-to-end Machine Learning pipeline, from raw data processing to a saved, deployable model.

**Developed by: Abdarhman Amin **

---

