# /r/houseplants - /r/gardening Classifier
## Subreddit Classification based on Post Titles

In this project, we address a critical challenge at PlantMa, our agricultural startup focused on selling home gardening supplies and plants. We aim to improve our marketing strategies by classifying Reddit posts from two distinct subreddits, /r/houseplants and /r/gardening, based on their post titles. The ultimate goal is to build a binary classification model that enables us to target our marketing efforts more effectively.

## Data Dictionary
| Feature        | Type    | Description                                            |
|----------------|---------|------------------------------------------------------- |
| title          | string  | The title of the Reddit post                           |
| subreddit      | integer | The source subreddit, 'houseplants': 1, 'gardening': 0 |


## Executive Summary
Garage Crafters Inc. embarked on a focused mission to provide expert garage renovation and construction services to homeowners in Ames, Iowa. The primary objectives of this project were to:

- Build a predictive model to evaluate the impact of specific features on home value/ sale price.
- Evaluate the impact of different types of garage renovations on the estimated property value.
- Identify specific neighborhoods in Ames in which new garage construction contributes to a high impact on home sale price.

To accomplish these goals, data related to neighborhoods, garage size, and garage type were analyzed. Machine learning techniques such as linear regression and ordinary least squares regression were employed to develop predictive models and uncover valuable insights.

The project provided Garage Crafters with data-driven recommendations for garage improvements, enabling homeowners to increase property value.

Data Source: https://www.kaggle.com/competitions/dsi-910-ames-housing-challenge/data

Original Data Compiled by Dean De Cock


### Data Cleaning and EDA (Exploratory Data Analysis)
Data cleaning primarily involved filling null values. Categorical null values were filled appropriately (e.g. no_pool). The following numerical values were filled with '0': 'Bsmt Full Bath', 'Bsmt Half Bath', 'Mas Vnr Area', 'Electrical', 'Garage Cars', 'Total Bsmt SF', 'Bsmt Unf SF', 'BsmtFin SF 2', 'BsmtFin SF 1', 'Garage Area'. A function 'cleaner' was created so that it could be applied to both the training and test data sets. 

EDA involved primarily making correlation heatmaps to quickly visually identify the correlation of variables to 'SalePrice'. It was necessary to create dummy columns for the categorical columns to be processed by seaborn. Pairplots from seaborn were also made to visualize correlations. Several sets of initial predictors were chosen, and further heatmaps were generated off these predictor sets.

### Preprocessing and Modeling
The following steps were performed during pre-processing and modeling:

- Focused EDA was performed to further explore garage-related features.

- Defined a list of predictors used for modeling. Predictors were chosen based on correlations and iterative evaluation of model performance. Relevant garage features were included.

- Split the training data into training and testing sets using an 80/20 split ratio.

- Selected categorical columns from the feature set and converted them into a list.

- Used one-hot encoding to transform the categorical columns and combined them with the remaining numerical columns.

- Built a linear regression model using the transformed training data and the target variable 'SalePrice'.

- Evaluated the model's performance by calculating the R-squared score on both the training and testing data, as well as the cross-validated R-squared score.

- Created a baseline prediction for the testing data by using the mean of the training data as the predicted value.

- Fitted an ordinary least squares (OLS) model to the training data using statsmodels.


### Evaluation and Interpretation
| Metric                                  | Score                |
|-----------------------------------------|----------------------|
| R-squared - Training Set               | 0.9093               |
| R-squared - Testing Set                | 0.8414               |
| Mean Cross-Validated Score (cv=5)      | 0.6512               |
| R-squared Score - Baseline             | -0.0149              |


The production model performed better than baseline. When evaluated on the training set, about 90.9% of the variability in Sale Price is captured by the predictor variables in the model. The model explains about 84.1% of the variability in sale price on the testing set, indicating that the model should generalize well to unseen data. The cross validation score indicates that, on average, the model explains about 65.1% of the variability in sale price across 5 subsets of data. 

Future further tuning of predictor features and hyper-parameters may improve the model. Note there is some high multicollinearity in the model. Most notably between 'Garage Cars' and 'Garage Area'. 

The following coefficients were recorded for business recommendations: 

| Variable                | Coefficient    |
|-------------------------|----------------|
| Neighborhood_GrnHill    | 120,400.00     |
| Neighborhood_StoneBr    | 37,920.00      |
| Neighborhood_NridgHt    | 28,290.00      |
| Garage Type_Attchd      | 40,310.00      |
| Garage Area             | 10.62          |
| Garage Cars             | 7,619.17       |

The following interpetrations are included for business recommendations:

Holding all else equal, the effect of having a garage in Green Hill, Stone Brook, or Northridge Heights neighborhoods contributes to an increase in house sale price of the following values respectively relative to not having a garage at all: \\$120,400.00, \\$37,920.00, \\$28,290.00.

Holding all else equal, the effect of having an attached garage on your house contributes to an increase in house sale price of \\$40,310.00.

Holding all else equal, for every 1 sq foot increase in garage area, we expect the sale price of the house to increase by \\$10.62.

Holding all else equal, for an increase in size of garage by 1 car, we expect the sale price of the house to increase by \\$7,619.17.
