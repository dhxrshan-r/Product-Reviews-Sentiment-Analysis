# Product Review Sentiment Analysis

A Natural Language Processing (NLP) and Machine Learning project that anaAlyzes customer product reviews and classifies them into **Positive, Neutral, and Negative** sentiment categories

The project uses **Sentence Transformers** to convert textual reviews into semantic embeddings, followed by classical Machine Learning classifiers such as **Random Forest** and **Gradient Boosting** for sentiment classification.

---

## 📌 Project Overview

Customer reviews contain valuable information about customer satisfaction, product quality, usability, and perceived value. However, manually analyzing a large volume of reviews is inefficient.

This project develops a sentiment classification framework that automatically analyzes product reviews and identifies their underlying sentiment.

### Objective

The main objectives are to:

* Classify product reviews into **Positive, Neutral, and Negative** categories.
* Generate meaningful text representations using **Sentence Transformer embeddings**.
* Train Machine Learning models on the generated embeddings.
* Compare model performance using **Accuracy** and **F1 Score**.
* Select the better-performing model for predicting sentiment on unseen reviews.
* Demonstrate how sentiment analysis can support data-driven business decisions.

---

## 🧠 Methodology

The project follows the workflow below:

```text
Product Reviews Dataset
        ↓
Data Loading & Inspection
        ↓
Data Cleaning
        ↓
Duplicate Removal
        ↓
Exploratory Data Analysis
        ↓
Sentence Transformer Embeddings
        ↓
Train-Test Split
        ↓
Machine Learning Models
   ┌───────────────┐
   │ Random Forest │
   │ Gradient Boost│
   └───────────────┘
        ↓
Model Evaluation
        ↓
Best Model Selection
        ↓
Unseen Review Prediction
        ↓
Sentiment Classification
```

---

## 📂 Dataset

The dataset contains **1,007 records and 3 columns** before duplicate removal.

### Features

| Column           | Description                                   |
| ---------------- | --------------------------------------------- |
| `Product ID`     | Unique identification number for each product |
| `Product Review` | Customer's written review of the product      |
| `Sentiment`      | Sentiment associated with the review          |

The notebook identifies **2 duplicate records**, which are removed before further analysis.

There are no missing values in the dataset.

### Sentiment Classes

* 🟢 **Positive**
* 🟡 **Neutral**
* 🔴 **Negative**

The dataset is imbalanced, with positive reviews forming the majority class and neutral/negative reviews occurring considerably less frequently.

---

## 🔍 Exploratory Data Analysis

The project performs basic EDA to understand the structure and distribution of the dataset.

The analysis includes:

* Dataset shape
* First few records
* Missing-value analysis
* Duplicate-value detection
* Sentiment distribution visualization

A count plot is used to examine the distribution of the three sentiment classes.

---

## 🤖 Sentence Transformer Embeddings

Instead of directly converting reviews into traditional numerical features, the project uses:

**`sentence-transformers/all-MiniLM-L6-v2`**

The Sentence Transformer converts each product review into a semantic embedding that captures contextual information from the text.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2'
)

embedding_matrix = model.encode(
    data['Product Review'].tolist(),
    show_progress_bar=True
)
```

These embeddings are then supplied to the Machine Learning classifiers.

---

## ⚙️ Data Preparation

The dataset is divided into training and testing sets using an **80:20 split**.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Stratification is used to maintain the sentiment-class distribution between the training and testing datasets.

---

## 🌲 Machine Learning Models

Two classification models are evaluated.

### 1. Random Forest + Transformer

A Random Forest classifier is trained using the Sentence Transformer embeddings.

```python
RandomForestClassifier(
    random_state=42,
    class_weight='balanced'
)
```

The `class_weight='balanced'` parameter is used to account for the imbalanced sentiment classes.

### 2. Gradient Boosting + Transformer

A Gradient Boosting classifier is also trained using the generated embeddings.

```python
GradientBoostingClassifier(
    random_state=42
)
```

---

## 📊 Model Evaluation

The models are evaluated using:

### Accuracy

Measures the overall proportion of correctly classified reviews.

### F1 Score

Measures the balance between precision and recall and is particularly useful when the dataset contains class imbalance.

---

## 📈 Model Comparison

According to the evaluation results reported in the notebook:

| Model                           | Test Accuracy | Test F1 Score |
| ------------------------------- | ------------: | ------------: |
| **Random Forest + Transformer** |    **~86.5%** |    **~81.8%** |
| Gradient Boosting + Transformer |        ~84.1% |        ~80.3% |

The notebook selects **Random Forest + Transformer** as the preferred model because it provides better test performance and lower generalization error compared with Gradient Boosting.

> **Note:** Some result values are reported slightly differently in separate notebook cells (for example, 86.1%, 86.5%, and 86.6% test accuracy). The README therefore uses the approximately reported values rather than presenting them as a single exact benchmark.

---

## 🔮 Prediction on Unseen Reviews

The selected Random Forest model is tested on **10 newly created product reviews** representing different sentiment patterns.

The workflow is:

```text
New Product Review
        ↓
Sentence Transformer
        ↓
Semantic Embedding
        ↓
Random Forest Classifier
        ↓
Predicted Sentiment
```

Example:

```python
unseen_predictions = rf_transformer.predict(
    unseen_embedding_matrix
)

unseen_df['Predicted Sentiment'] = unseen_predictions
```

The predictions are then displayed and visualized using a sentiment distribution chart.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries & Frameworks

* **Pandas** – Data manipulation
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization
* **Seaborn** – Exploratory visualization
* **Scikit-learn** – Machine Learning and evaluation
* **Sentence Transformers** – Text embeddings
* **Imbalanced-learn** – Imbalanced-data utilities

### Machine Learning

* Random Forest Classifier
* Gradient Boosting Classifier
* Train-Test Split
* F1 Score
* Accuracy Score

### NLP

* Sentence Embeddings
* Semantic Text Representation
* Product Review Sentiment Classification

---

## 📁 Project Structure

```text
Product-Review-Sentiment-Analysis/
│
├── Product_Review_Sentiment_Analysis.ipynb
├── Product_Reviews.csv
└── README.md
```

> If the dataset is not included in the repository, remove `Product_Reviews.csv` from the structure and provide instructions for obtaining or placing the dataset locally.

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/dhxrshan-r/Product-Review-Sentiment-Analysis.git
```

### 2. Navigate to the Project

```bash
cd Product-Review-Sentiment-Analysis
```

### 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn sentence-transformers imbalanced-learn
```

### 4. Open the Notebook

Run the notebook using:

* **Google Colab**
* **Jupyter Notebook**
* **JupyterLab**

### 5. Update the Dataset Path

The notebook currently loads the dataset from a Google Drive path:

```python
data = pd.read_csv(
    '/content/drive/MyDrive/GenAI_Class/Dataset/Product_Reviews.csv'
)
```

For GitHub/local execution, update this path according to the location of your dataset.

---

## 📌 Key Takeaways

* Product reviews can be transformed into meaningful semantic representations using Sentence Transformers.
* Classical Machine Learning models can effectively classify transformer-generated embeddings.
* Random Forest performed better than Gradient Boosting in the experiments documented in this notebook.
* The approach can be extended to larger and continuously updated review datasets.
* Sentiment classification can provide actionable insights for product and customer-experience improvement.

---
