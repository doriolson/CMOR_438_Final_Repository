# Datasets for CMOR 438 Machine Learning Projects

This folder contains the datasets used throughout this repository for both supervised and unsupervised machine learning experiments. Both datasets are sourced from **Kaggle** and have been chosen for their richness, diversity, and suitability for a variety of machine learning tasks, including classification, regression, clustering, and community detection.

---

## 1. Supervised Learning Dataset: Income Dataset

**File:** `Income_Dataset.csv`  

The Income Dataset provides demographic and socioeconomic information for a large set of individuals. Each observation includes features such as age, education, gender, marital status, occupation, work hours, and capital gains or losses. The dataset also contains the target variable `income`, which indicates whether an individual earns above or below a specified threshold (e.g., \$50K annually).

This dataset is ideal for supervised learning because it combines **numeric and categorical features**, allowing exploration of encoding strategies, feature scaling, and model evaluation techniques. It can be used for classification tasks, regression approximations, and testing the interpretability of models such as decision trees, ensemble methods, linear regression, logistic regression, and neural network-based approaches.  

**Variable Definitions:**

- `age`: Age of the individual in years.  
- `workclass`, `occupation`, `marital-status`: Categorical descriptors of employment and marital status.  
- `education`, `educational-num`: Education level in categories and numeric encoding.  
- `hours-per-week`: Number of hours worked per week.  
- `capital-gain`, `capital-loss`: Financial gain or loss from investment activities.  
- `income`: Target variable indicating income level.  

The dataset has been preprocessed to handle categorical encoding, and it contains no missing values, making it straightforward for modeling and evaluation.

---

## 2. Unsupervised Learning Dataset: Obesity Dataset

**File:** `Obesity_Dataset.csv`  

The Obesity Dataset provides information on individual lifestyle habits, dietary patterns, and physical measurements. It includes numeric features such as `age`, `height`, `weight`, and consumption-based measures (`FCVC`, `NCP`, `CH2O`, `FAF`, `TUE`), as well as categorical variables describing habits like `CALC` (consumption of high-calorie foods), `FAVC` (frequency of high-calorie food intake), `SCC` (smoking status), `SMOKE`, family history of overweight, and transportation modes (`MTRANS`). The dataset also includes `NObeyesdad`, which classifies obesity levels for reference but is not used as an input in unsupervised modeling.  

This dataset is particularly well-suited for unsupervised learning techniques because it allows exploration of clustering, density-based algorithms, and community detection. Its combination of numeric and categorical features permits normalization, encoding, and graph-based analysis. The dataset supports a variety of analytical approaches, from PCA visualization and K-Means clustering to DBSCAN and community detection.

**Variable Definitions:**

- `Age`, `Height`, `Weight`: Basic demographic and physical measurements.  
- `FCVC`: Frequency of vegetable consumption.  
- `NCP`: Number of main meals per day.  
- `CH2O`: Daily water consumption in liters.  
- `FAF`: Physical activity frequency.  
- `TUE`: Time using electronic devices.  
- `CALC`, `FAVC`, `SCC`, `SMOKE`: Lifestyle habits and behaviors.  
- `family_history_with_overweight`: Indicates presence of overweight family members.  
- `CAEC`: Consumption of high-calorie food.  
- `MTRANS`: Mode of transportation.  
- `NObeyesdad`: Obesity classification (used only for evaluation).  

The dataset contains no missing values and provides a moderate number of observations (2,111) and features (17), making it suitable for clustering and community detection experiments.

---

Both datasets provide rich, real-world examples for applying a wide range of supervised and unsupervised machine learning algorithms, allowing experimentation with preprocessing, modeling, visualization, and evaluation techniques.
