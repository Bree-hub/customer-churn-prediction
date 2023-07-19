# Predicting Customer Churn for SyriaTel Telecommunications Company.

* Student name: GROUP 9
* Members: 
 1. Peggy Obam
 2. Rael Ndonye
 3. Alan Omondi
 4. Jackson Maina
 5. Jared Bii
 6. Brenda Kinoti

* Student pace: part time  



## Background

Customer churn is one of the most important factors to consider in measuring any company's success or growth. By definition, customer churn refers to the percentage of customers that stopped using your product or service during a certain period of time. SyriaTel Telecommunications Company in a bid to measure its customer churn, identified the causes. Therefore using machine learning algorithmns, we can give the most accurate predictions for future predictions.


## Business Problem

In an ideal environment, every business would retain all of its customers. However, customer attrition is a reality that all businesses must face. SyriaTel Telecommunications Company has taken steps to address this issue by identifying the causes of customer attrition. Therefore we can use machine learning algorithms to predict which customers are most likely to churn. This information can be used to target customers with specific interventions to prevent them from churning.

The primary objectives of this project are as follows:

a. To Build a classification model to predict customer churn for SyriaTel.

b. To Identify the key factors influencing customer churn.

c. To Provide insights and recommendations to SyriaTel for effective churn management.


## Data
For this project, we have chosen the "SyriaTel Customer Churn" dataset. The dataset provides various customer-related information such as 'state', 'account length', 'area code', 'phone number', 'international plan', 'voice mail plan', 'number vmail messages', and several other features related to call duration, charges, and customer service interactions. This suggests that the dataset covers a wide range of customer attributes. 

This dataset is particularly suitable for our objectives, as it provides the necessary information to understand customer behavior and predict churn.

SyriaTel Customer Churn" dataset has 3333 rows and 21 columns.The dataset contains data including: 
1. daytime calls, minutes and pricing.
2. night time calls, minutes and pricing.
3. International calls, minutes, pricing
4. Customer service calls
5. If customer had International plan
6. Customer churn.


## Methods

The project will follow the following steps:

a. Exploratory Data Analysis: We will perform an in-depth exploration of the dataset to gain insights into the distribution of variables, identify patterns, and detect any data quality issues.

b. Data Preprocessing: This step involves handling missing values, encoding categorical variables, and scaling numerical features.

c. Feature Selection: We will identify relevant features that have a significant impact on customer churn prediction. 

d. Model Selection and Training: We will compare various classification algorithms, such as logistic regression, decision trees, and random forests, to select the most suitable model for predicting customer churn. 

e. Model Evaluation: We will assess the performance of the trained model using appropriate evaluation metrics, including accuracy, precision, recall, and F1-score. 

f. Model Optimization: We will fine-tune the selected model by adjusting hyperparameters and employing techniques like grid search. This optimization process aims to maximize the model's predictive capabilities. The models used include Logistic Regression, KNN, Decision Trees and Random Forests. 

In each model the performance metrics; accuracy, precision, recall and f1 score were calculated. Confusion matrix for each model was also plotted. The best model is then evaluated from the four models. For Decision Trees only, feature importance to see which features played a role in customer churn is provided.


## Exploratory Data Analysis
## Univariate analysis
Sample univariate analysis visualization:

1. Distribution of the target variable 'churn'
   
![image](https://github.com/JaredBii/Phase-3-project-Customer-Churn-Prediction-for-SyriaTel/assets/29143340/09036c5c-fefc-44bd-8e47-87c654f62178)

2. Distribution and potential outliers of the 'customer service calls' variable:

![image](https://github.com/JaredBii/Phase-3-project-Customer-Churn-Prediction-for-SyriaTel/assets/29143340/6ba52c02-8de9-4622-b2bb-54a8ef35b33c)

3. Frequency of customers with and without the 'international plan':

![image](https://github.com/JaredBii/Phase-3-project-Customer-Churn-Prediction-for-SyriaTel/assets/29143340/f41cf7a0-55b3-4848-96b1-4cb66b03f71b)

4. Distribution Plots for all features:

![image](https://github.com/JaredBii/Phase-3-project-Customer-Churn-Prediction-for-SyriaTel/assets/29143340/2bfdb34c-ae9c-4804-9e48-1338079b201a)

## Bivariate analysis
Sample bivariate analysis done:

1. Correlation matrix


![image](https://github.com/JaredBii/Phase-3-project-Customer-Churn-Prediction-for-SyriaTel/assets/29143340/8e147bb5-f3cd-4b20-b5d5-bb3e820b74d8)


## Modelling

### Logistic Regression Model
Logistic regression is a regression analysis technique that is specifically designed for situations where the dependent variable is categorical and can only take discrete values.

#### Observations 
1. Accuracy: The model's accuracy is 85.46%, indicating the percentage of correctly predicted instances. 

2. Precision: The precision is 56.25%, implying that only half of the predicted positive instances are actually true positives.

3. Recall: The recall is 17.83%, indicating the model's ability to correctly identify positive cases among all actual positive cases.

4. F1 Score: The F1 score, at 0.270677, represents a moderate balance between precision and recall.

5. AUC Score: The AUC score of 0.80338 suggests reasonable discrimination ability in distinguishing between positive and negative instances.

Overall, the observations reveal limitations in correctly identifying positive instances (low recall) and achieving a balanced precision and recall (low F1 score). Further analysis and model refinement may be necessary to enhance performance.

![image](https://github.com/JaredBii/Phase-3-project-Customer-Churn-Prediction-for-SyriaTel/assets/29143340/4919ba18-4e2d-40e6-ab61-6a0716c033b2)



###  K-Nearest Neighbors
The k-nearest neighbors (KNN) algorithm is a supervised machine learning method that estimates the probability of a data point belonging to a particular group by considering the group memberships of its nearest neighboring data points.

#### Observations

1. Accuracy: The accuracy of 0.855 indicates that approximately 85.5% of the instances in the evaluation dataset were correctly classified by the KNN model. 

2. F1 Score: The F1 score of 0.076 is a measure that balances both precision and recall. It indicates the harmonic mean of these two metrics. A low F1 score suggests poor performance in terms of correctly identifying positive instances and minimizing false positives.

3. Precision: The precision of 1.000 suggests that all instances predicted as positive by the KNN model were true positives. However, it is crucial to examine other metrics to assess the overall performance of the model.

4. Recall: The recall of 0.040 indicates that only a small proportion (approximately 4%) of actual positive instances were correctly identified by the KNN model. 

5. ROC AUC Score: The ROC AUC score of 0.520 represents the Area Under the Receiver Operating Characteristic Curve (ROC AUC). 

Overall, the observations indicate that the KNN model may have limitations in correctly identifying positive instances (low recall), and its overall performance in terms of precision, recall, and discrimination ability is relatively poor. Further analysis and model refinement may be necessary to improve its performance.

![image](https://github.com/JaredBii/Phase-3-project-Customer-Churn-Prediction-for-SyriaTel/assets/29143340/8affab65-a468-44a5-899e-527c96a7beb5)



### Decision Trees
Decision Trees (DTs) are a type of supervised learning technique used for classification and regression tasks. The objective is to build a model that can predict the value of a target variable based on simple decision rules learned from the features present in the data.

#### Observations

1. Accuracy: The accuracy of 0.93 indicates that approximately 93% of the instances in the evaluation dataset were correctly classified by the decision tree model. 

2. F1 Score: The F1 score of 0.761 is a measure that balances both precision and recall. It represents the harmonic mean of these two metrics.

3. Precision: The precision of 0.781 suggests that around 78.1% of the instances predicted as positive by the decision tree model are actually true positives. 

4. Recall: The recall of 0.743 indicates that approximately 74.3% of the actual positive instances were correctly identified by the decision tree model.

5. ROC AUC Score: The ROC AUC score of 0.853 represents the Area Under the Receiver Operating Characteristic Curve (ROC AUC). This metric assesses the model's ability to distinguish between positive and negative instances.

Overall, the observations suggest that the decision tree model performs well in terms of accuracy, precision, recall, F1 score, and discrimination ability. However, further analysis and validation with additional evaluation metrics may be necessary to gain a more comprehensive understanding of the model's performance.

![image](https://github.com/JaredBii/Phase-3-project-Customer-Churn-Prediction-for-SyriaTel/assets/29143340/9bb6d414-9247-4175-a1bc-b3b30edc7ad7)


### Random Forests
Random Forests are machine learning algorithm used for both classification and regression tasks. It combines the predictions of multiple individual models to make final prediction.

#### Observations

1. Accuracy: The accuracy of 0.919 indicates that approximately 91.9% of the instances in the evaluation dataset were correctly classified by the random forest model.

2. F1 Score: The F1 score of 0.635 is a measure that balances both precision and recall. It represents the harmonic mean of these two metrics. 

3. Precision: The precision of 1.00.This metric measures the accuracy of positive predictions.

4. Recall: The recall of 0.465 indicates that approximately 46.5% of the actual positive instances were correctly identified by the random forest model. This metric evaluates the model's ability to find all positive instances.

5. ROC AUC Score: The ROC AUC score of 0.733 represents the Area Under the Receiver Operating Characteristic Curve (ROC AUC). 

Overall, the observations suggest that the random forest model performs well in terms of accuracy and precision, indicating good overall predictions and accurate positive classifications. However, the model's performance in terms of recall and F1 score is relatively lower, suggesting room for improvement in correctly identifying positive instances and achieving a better balance between precision and recall.

![image](https://github.com/JaredBii/Phase-3-project-Customer-Churn-Prediction-for-SyriaTel/assets/29143340/a769d165-32e0-4f8a-8e4b-2b927b9c0671)


## Results

### Model Selection

1. Accuracy: The Decision Tree and Random Forest models perform similarly well with accuracies of 0.930 and 0.919, respectively. Logistic Regression and KNN have slightly lower accuracies.

2. Precision: Decision tree achieves the highest precision score of 0.781, indicating a high proportion of correct positive predictions. Logistic Regression has relatively lower precision score, while KNN and Random Forest achieves a perfect precision score of 1.000.

3. Recall: The Decision Tree model has the highest recall score of 0.743, indicating its ability to identify a higher proportion of positive instances. Logistic Regression and Random Forest have relatively lower recall scores, while KNN performs the poorest in terms of recall.

4. F1 Score: The Decision Tree model has the highest F1 score of 0.761, which considers both precision and recall. Random Forest follows. Logistic Regression and KNN have lower F1 scores, with KNN having the lowest.

5. ROC AUC Score: The Decision Tree model achieves the highest ROC AUC score of 0.853, indicating its better ability to distinguish between positive and negative instances. Random Forest and Logistic Regression have relatively lower ROC AUC scores, while KNN has the lowest score.

In summary, the Decision Tree and Random Forest models generally perform better across multiple evaluation metrics, including accuracy, precision, recall, F1 score, and ROC AUC score. Logistic Regression performs moderately, while KNN shows relatively lower performance in most of the evaluation metrics.

![image](https://github.com/JaredBii/Phase-3-project-Customer-Churn-Prediction-for-SyriaTel/assets/29143340/e23d80e1-f0fd-48e4-b589-73ea54b7437b)


### Feature Importance
1. The most important feature for predicting churn is **total day charge**, which has a score of 0.268983. This means that the amount of money a customer spends on day calls is a strong predictor of whether they will churn.

2. The second most important feature is **customer service calls**, which has a score of 0.113732. This means that customers who make more customer service calls are more likely to churn.

3. Other important features include **total eve charge**, **total intl charge**, and **international plan_yes**. These features all relate to the amount of money a customer spends on their phone service, which is a strong predictor of churn.

4. The least important features are **account length**, **area code**, and **number vmail messages**. These features do not seem to be very predictive of churn.

Overall, the feature importance indicates that the amount of money a customer spends on their phone service is a strong predictor of whether they will churn. Other important factors include the number of customer service calls a customer makes and whether they have an international plan.

![image](https://github.com/JaredBii/Phase-3-project-Customer-Churn-Prediction-for-SyriaTel/assets/29143340/2655d370-c458-4d13-b312-7107a86c3d82)


## Conclusions

Decision Tree model appears to be the best performer among the four models. This would be the best Model for the Syria Tel Telecommunication Company to use to predict which customer will unsubscribe from their services and take precautionary steps to reduce the churn rate.

The Most important features for predicting churn are:
- Total day charge
- Customer Service call
- Total eve charge
- Total intl charge



## RECOMMENDATIONS


* **Focus on reducing the amount of money customers spend on day calls.** This is the most important factor in predicting churn, so it is the most important area to focus on. This could be done by offering discounts on day calls, or by providing customers with more affordable alternatives to day calls.

* **Reduce the number of customer service calls.** Customers who make more customer service calls are more likely to churn. This could be done by improving the customer service experience, or by making it easier for customers to resolve their issues without having to call customer service.

* **Consider offering international plans.** Customers who have international plans are less likely to churn. This could be done by offering more affordable international plans, or by making it easier for customers to sign up for international plans.

* **Ignore account length, area code, and number email messages.** These features are not very predictive of churn, so there is no need to focus on them.
