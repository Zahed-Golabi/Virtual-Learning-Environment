# Virtual Learning Environment

<img src="./vle.jpg" alt="VLE" title="Virtual Learning Environment">

---

## Table of Contents

1. How to run the project
2. Data preprocessing
3. Data Visualization
4. Training Models
---
### 1. Running the project
In order to run the project, do the followings:<br>
- Go to the root directory of the project
- Run ```python3 -m venv .venv```<br>
- Then ```source .venv/bin/activate```<br>
- Finally, ```pip install -r requirements.txt```<br>
- To get all results, run ``` python prediction.py```
---
### 2. Data preprocessing
- Functions:
  - replace_values
  - fill_values
  - feature_extraction
  - drop_features
  - save_new_dataset
  - feature_encoding
---
### 3. Data visualization
- Functions:
  - stats
  - feature_distribution_plot
  - feature_correlation_barplot
  - feature_correlation_scatterplot
  - feature_correlation_catplot
  - display
  - feature_importance_plot
  - model_evaluation_plot
---
### 4. Model training
- Functions
  - fit_binary
  - fit_multiclass
---
