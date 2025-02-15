# Predicting Schizophrenia based on FNC and SBM values

This project focuses on predicting whether a patient has Schizophrenia or not binary classificaiton. The dataset is  **high-dimensional**, with **86** rows and **412** features. To tackle challenges with the data, i applied several data preparation techniques, evaluated pearson corrleation, visualized the data, and trained random forest model with grid search hyperparameterr tuning to get the best of the model to achieve high predictive accuracy.

---

## Overview

Due to the **large dimensionality (412 features)** and **complex, uninformative feature names**, straightforward inspection of the dataset is not feasible. The numeric columns provide little immediate insight but applying advanced data manipulation and exploration techniques was necessary.

---

## Key Steps

### 1. Finding High linearly correlated features
I applied a filtering process where i can spot  highly correlated features because we have more than 412 of them, after finding them i visualized the outliers and could not foind one with outliers. 

---

### 2. Data Visualziation
I used matplotlib ad seaborn libraries to create visualizations for the data. I utilized plt.subplots() method to create a grid for the plots and used seaborn to plot the plots. 

![Plots](screenshots/plot_w_sns_plt.png)

---

### 3. QuantileTransformer for Scaling

## 4. Deep Learning Model
A deep learning architecture was implemented to handle the high-dimensional feature space. Below are snapshots of the model’s structure and the training progression:

### Model Architecture
![Linearity](screenshots/1_model.png)
![Linearity](screenshots/2_model.png)
---

### Initial Epochs
The model converged relatively quickly:
![Linearity](screenshots/initial_epochs.png)
---

### Final Epochs
By the end of training, the model achieved nearly **95% accuracy**:
![Linearity](screenshots/final_epochs.png)
---

## 5. Prediction Results
After data preprocessing, balancing, and scaling, our deep learning model demonstrated strong predictive performance on the test set. These results highlight the effectiveness of the chosen techniques.

---

## Conclusion
In this project, I tackled a **high-dimensional, imbalanced** dataset by:

- Evaluating linear vs. non-linear relationships  
- Applying SMOTE for class balancing  
- Using Quantile Transformation for scaling  
- Training a deep neural network  

These steps allowed us to extract meaningful insights and achieve an accuracy of nearly **92%*
