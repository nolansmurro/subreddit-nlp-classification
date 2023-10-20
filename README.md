# Binary Classification Models: /r/houseplants - /r/gardening 
## Subreddit Classification based on Post Titles

## Overview

PlantMa, our agricultural startup company specializing in houseplants and gardening supplies, is looking to optimize its marketing strategies and product offerings for two distinct customer segments: houseplant enthusiasts and gardening enthusiasts. We aim to improve our marketing strategies by building a model to classify Reddit posts from two distinct subreddits, /r/houseplants and /r/gardening, based on their post titles. The ultimate goal is to build a classification model that provides insights enabling us to target our marketing efforts more effectively and could potentially be repurposed for use in other online communities.

## Data Dictionary
| Feature       | Type    | Description                                             |
|---------------|---------|---------------------------------------------------------|
| id            | string  | Unique identifier for the Reddit post                   |
| created_utc   | integer | The timestamp of when the Reddit post was created      |
| title         | string  | The title of the Reddit post                            |
| self_text     | string  | The text content of the Reddit post                     |
| comments      | integer | The number of comments on the Reddit post               |
| score         | integer | The score (upvotes - downvotes) of the Reddit post      |
| upvote_ratio  | float   | The ratio of upvotes to the total votes for the post    |
| subreddit     | integer | The source subreddit: 'houseplants' (1) or 'gardening' (0) |

## Executive Summary

Our project's primary objectives are as follows:
- Collect and analyze Reddit posts to gain insights into the distinct characteristics of the "houseplants" and "gardening" subreddits by the end of the fiscal quarter.
- Develop a classification model that accurately predicts the subreddit (houseplants or gardening) based on post text content with at least 80% accuracy on testing datasets.
- Provide PlantMa with actionable insights and recommendations based on our analysis to enhance marketing and product strategies.

To accomplish these goals, data was scraped from Reddit using the PRAW library. After data cleaning and EDA, binary classification models were created utilizing logistic regression and extra tree classifiers.

The project provided valuable data-driven insights to PlantMa which will aid in their marketing campaigns. 

Data Sources:
- www.reddit.com/r/houseplants/
- www.reddit.com/r/gardening/

## Data Collection
The PRAW (Python Reddit API Wrapper) Library was utilized to collect a total of 12,052 posts from /r/houseplants and /r/gardening utilizing the following filters:
- new
- hot
- rising
- top (all, year, month, week)

The following features were retrieved during data collection:
- id
- created_utc
- title
- self_text
- comments
- score
- upvote_ratio
- subreddit

This data was saved to two csv files for cleaning: 'houseplants.csv' and 'gardening.csv'.

## Data Cleaning
Data cleaning involved handling null values and making the dataset suitable for analysis:
- Imported houseplants and gardening csv files as individual dataframes
- Concatenated both dataframes together
- Dropped 2781 duplicate rows.
- Filled null values appropriately
- Dummified the subreddit column to create a binary target variable (1 for "houseplants" and 0 for "gardening")


## EDA (Exploratory Data Analysis)
EDA was performed to gain insights into the data:
- Examined correlations between features such as post comments, scores, and upvote ratios
- Employed count vectorization to discover the most frequently used words in both subreddits

<img src="images/top_features_gardening.png" alt="Top Features Gardening" width="500">
<img src="images/top_features_houseplants.png" alt="Top Features Houseplants" width="500">

## Modeling

### Model 1: Count Vectorization / Logistic Regression
Model 1 utilized count vectorization in conjunction with the logistic regression classifier.

#### Data Loading: 
The dataset 'plants.csv' was loaded into a Pandas DataFrame named 'plants.' The 'title' column was defined as the feature ('X'), while the 'subreddit' column was used as the target variable ('y').

#### Train-Test Split: 
The dataset was split into training and testing sets using train_test_split, reserving 25% of the data for testing.

#### Custom Stop Words: 
A list of custom stop words was created, which included words like 've,' 'help,' and 'little,' in addition to common English stop words. All stop words were combined into the 'all_stop_words' list.

#### Benchmark Model: 
A benchmark model was set up with Count Vectorization (cvec_bench) and Logistic Regression (log_reg_bench) using a simple pipeline. It was trained on the training data, and its accuracy on both the training and testing sets was calculated.

#### Tuned Model: 
A more sophisticated tuned model was created using a pipeline with Count Vectorization and Logistic Regression. A grid search (GridSearchCV) was performed to identify the best hyperparameters through cross-validation.
|   Hyperparameter  | Model 1 Value |
|-------------------|---------|
| cvec max df       | 0.9     |
| cvec min df       | 1       |
| cvec max features | None    |
| cvec n gram range | (1,2)   |
| cvec stop words   | English + Custom |

#### Model Evaluation: 
The model's performance was assessed by calculating and displaying the accuracy scores on the training and testing sets. Confusion matrices were generated to visualize true positives, true negatives, false positives, and false negatives. A classification report was printed, providing precision, recall, F1-score, and support for each subreddit class. Additionally, the model identified and displayed misclassified posts, along with their true and predicted labels.

<img src="images/model_1_confusion.png" alt="Model 1 Confusion Matrix" width="500">

#### Model Results:
|      Metric    | Model 1 Score |
| ------------- | ------- |
| Baseline      | 0.489   |
| Benchmark Training | 0.937 |
| Benchmark Testing  | 0.837 |
| Tuned Training  | 0.977   |
| Tuned Testing   | 0.846   |

##### Interpretation:
The model achieved a high training accuracy of 97.74%, suggesting strong performance on the training data. However, the significant gap between training and testing accuracy indicates a potential issue of overfitting.
The testing accuracy of 84.69% shows reasonable generalization to new, unseen data.

Addressing overfitting may involve adjusting model complexity, regularization, or dataset size.
Further hyperparameter tuning may optimize testing data performance.
Further text preprocessing may also increase model performance. Self-text and numerical features may be considered in future model iterations.

### Model 2: Count Vectorization/ Extra Trees Classifier
In the second model, Count Vectorization was used in conjunction with the Extra Trees classifier.

#### Data Loading:
The cleaned dataset 'plants.csv' was loaded into a Pandas DataFrame named 'plants.' The 'title' column was defined as the feature ('X'), while the 'subreddit' column was used as the target variable ('y').

#### Train-Test Split: 
The data was divided into training and testing sets using train_test_split, reserving 25% of the data for testing.

#### Benchmark Model: 
The model set up a benchmark with Count Vectorization (cvec_bench) and the Extra Trees Classifier (et_bench) using a pipeline. The benchmark model was trained on the training data, and its accuracy on both the training and testing sets was calculated.

#### Tuned Model: 
The tuned model was created using a pipeline that combined Count Vectorization and the Extra Trees Classifier. A grid search (GridSearchCV) was performed to find the best hyperparameters using cross-validation.
|     Hyperparameter  | Model 2 Value|
|---------------------|---------|
| cvec max df         | 0.9     |
| cvec min df         | 1       |
| cvec max features   | None    |
| cvec n gram range   | (1,2)   |
| cvec stop words     | None    |
| et max depth        | None    |
| et max features     | sqrt    |
| et min samples leaf | 1       |
| et n estimators     | 500     |

#### Model Evaluation: 
The model's performance was evaluated by calculating and displaying the accuracy scores on the training and testing sets. Confusion matrices were generated to visualize true positives, true negatives, false positives, and false negatives. A classification report was printed, providing precision, recall, F1-score, and support for each subreddit class. Additionally, the model identified and displayed misclassified posts, along with their true and predicted labels.

<img src="images/model_2_confusion.png" alt="Model 2 Confusion Matrix" width="500">

#### Feature Importance: 
The model calculated and visualized the feature importances from the best estimator. The top 20 most important features were displayed in a horizontal bar chart.

<img src="images/model_2_feature_importance.png" alt="Model 2 Feature Importance" width="500">

#### Model Results:
|      Metric   | Model 2 Score |
| ------------- | ------- |
| Baseline      | 0.489   |
| Benchmark Training | 0.995 |
| Benchmark Testing  | 0.839 |
| Tuned Training  | 0.995   |
| Tuned Testing   | 0.845   |

##### Interpretation:
During training, the model achieved extremely high accuracy of approximately 99.54%. It correctly classified almost all the posts in the training dataset into their respective subreddits. However, such a high training accuracy may also indicate overfitting.

For this model, the testing accuracy was approximately 84.51%. Despite the high training accuracy, the model's generalization to new, unseen data was lower, suggesting that it may be overfitting the training data.


### Tabulated Result Comparison
| Model                    | Baseline | Benchmark Training | Benchmark Testing | Tuned Training | Tuned Testing |
|--------------------------|----------|--------------------|-------------------|---------------|--------------|
| Model 1, cvec/log_reg    | 0.489    | 0.937              | 0.837             | 0.977         | 0.846        |
| Model 2, cvec/extra trees| 0.489    | 0.995              | 0.839             | 0.995         | 0.845        |

### Evaluation
In summary, while Model 1 and Model 2 score very similarly on the testing sets, the logistic regression model is also a 'whitebox' model and is therefore more interpretable. It is our reccomendation to utilize Model 1 as the production model for this reason.

