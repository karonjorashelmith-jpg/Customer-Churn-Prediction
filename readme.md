# Customer Churn Prediction(Telcom Industry)
![](/churn%20images%20-%20Copy/running%20photo.jpeg)
## Introduction
Customer churn is when subscibers stop using services being provided by a firm. When customers leave, the firm loses revenue and are required to spend additional resources to attract new customers. Predicting churn before it happens becomes a vital thing for a company which helps them take action early.
The goal of the project is to build a machine learning mdel that can predict whether a customer is at a high risk of churn.

### Objectives
- To explore the data and find which features contribute greatly to customer churn.
- To fnd the percentage of customers that do not churn and predict those who are at risk of churning.
- Find a suitable machine learning model to classify chur and -not churn.


### Importance

- To reduce loss of revenue
- Deploy resources targeted at customer retention srategies.
- The business is able to understand which services influence churn.

### Dataset
<https://www.kaggle.com/datasets/blastchar/telco-customer-churn>

The dataset has information like:-

- customer with churn and those who stay
- customers are subscribed to e.g tech support, multiple lines
- customer information e.g contract, dependents, payment method

## Data loading, inspection and handle missing values
- Load CSV file  into pandas 
-inspect columns, datatypes and missing values:- df.info(), df.isna().sum()
- Convert Totalcharges into numericals and handle the missing values.
-Standardize text values like (yes/no)



## Exploratory Data Analysis
### 1. Churn distribution
Churn and not churn distribution - 26.5% of customers switched to another firm. This also shows about 73.5% of customers remain with the company , showing healthy retention

![Churn Distribution Pie Chart](images/churn%20distribution%20pie%20chart.png)


### 2. Churn distribution by gender
Churn distribution with respect to gender - Analysis of churn with respect to  gender shows there is negligible difference of churn rates between male and female. This means that gender is not a strong predictor of churn behaviour. Gender distribution seems balanced and healthy customer retention is evident.

![Churn by Gender](images/churn%20by%20gender.png)

### 3. Customer churn by contract type
Customer churn by contract type - customers with month-to-month subscription have the highest churn compared to one and two year countries. 

![](/churn%20images/churn%20by%20contract.png)

### 4. Internet services vs churn
 
 Customers using fiber optics churn the most but they also have the largest customer base. This could indicate that customers are not happy with the internet services. Customers using DSL churn less than the fiber optic which indicates more satisfaction with the service.

![Churn Distribution Pie Chart](images/churn%20distribution%20pie%20chart.png)

 ### 5. Payment method vs churn

 Payment method vs churn -The barchart shows that the Electronic check customers, churn the most while automatic pay churn the least. This group should be targeted when it comes to retention strategies.

 ![Payment Method by Churn](images/payment%20method%20by%20churn.png)

 ### 6. Tech suppport vs churn
  Customers with no TechSupport churn the most. They move to another provider where they feel supported.In contrast, customers with TechSupport showed a significantly lower churn rate. Availability of technical support play a big role in cusomer retention.

![Churn by Tech Support](images/churn%20by%20tech%20support.png)

### 7. Online Security vs churn
Customers with no online security churn the most.

![Churn by Online Security](images/churn%20by%20online%20security.png)

### 8. Paperless billing vs churn

 Paperless billing is associated with higher churn risk which makes it a meaningful predictor of churn.

![Paperless Billing by Churn](images/paperless%20billing%20by%20churn.png)

### 9. Tenure vs churn
Customers that churn more are those with  short tenures(months) while those with long term tenores seem loyal to the firm.

![](/churn%20images/churn%20by%20tenure%20distribution.png)

### 10. Monthly charges vs churn
Customers that leave pay more monthly charges compared to those who stay. The median monthly charge for churn customers is about $80 while for those who stay its about $65.

![Churn by Tenure Distribution](images/churn%20by%20tenure%20distribution.png)

### 11.
Total charges vs churn - The distribution is right skewed which means that most customers have low total charges and few customers have high total charges.
![Density Distribution of Total Charges](images/density%20distribution%20of%20total%20charges.png)


## Feature Engineering
New features created include:-
 - Spending Rate
 - Tenure Group
 - Num Services
 - Binary mappings for yes/no columns(PaperlessBilling,dependents, payment method)
 - new vs long term customers

This is to improve model accuracy, highlight customer behaviour, Reduce noise to improve model stability

The barchart below shows the number of services customers subscribe to and those that churn. Those with few services churn the most.

![Churn by Number of Services](images/churn%20by%20num%20services.png)

## Feature selection
### Chi-square

The strongest predictors were contract, online security,tech support,tenure group and Paperless Billingwhich had the lowest p-value of < 0.05

### VIF
Why:- To detect multicollinearity

In thisdataset, many features were related to each other. For example:   
 - MonthlyCharges and NumServices tend to increase together

 - Tenure and LongTermCustomer convey similar information

- Spending_Rate is derived from other variables.---This variable was dropped.

### Encoding
#### Label-encode binary features (Yes/No → 1/0).
Binary features with yes/no converted to 0s and Is 

    e.g 
    Partner 
    Dependents
    Paperless Billing

#### One-Hot Encoding
Other Categorical variables like contract,internet services were converted into new columns for each category.

*drop_first*=True was used to avoid dummy variable trap.


# Machine Learning Model Evaluations and Predictions:

## Train/test split
train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

### Logistic Regression

Confusion Matrix:
 [[940  95]
 [179 195]]

 ![Logistic Regression](images/logistic%20regression.png) 

The model can correctly predict customer churn 80.5% of the time. the ROC-AUC is 0.846 which mean it does an excellent job separating the churners from non-churners.

Non-churners were correctly identified  with high recall (91%), but performs moderately on detecting churners (recall = 52%), and misses churners 48% of the time.

Although logistic regression provides a solid baseline, it misses some churners (false negatives). Therefore, a more advanced model such as Random Forest is recommended to improve churn detection.


### Random forest

        Precision  recall  f1-score  support 
         0.82       0.89    0.85      1035     0
         0.59       0.46    0.52       374     1
Accuracy: 0.772888573456352

Confusion Matrix:
 
 [[917 118]
  
 [202 172]]

 ![Random Forest](images/random%20forest.png)

The Random forest model correctly predicted churn/no churn 77% of the time. Although the model performed reasonably well, its ability to identify customers who are actually likely to churn was limited. 54% of the customers that left the model missed them. 

The confusion matrix shows that the model correctly identified 172 of all churners and misclassified 202 churners as non-churners. 917 non-churners were correctly identified and 118 non-churners were incorrectly flagged as churn risk.

The model’s ROC-AUC score of 0.819 indicates fairly good discrimination between churners and non-churners. While the score shows that the model can separate the two groups better than chance, there is still room for improvement, especially in detecting churners more consistently.







