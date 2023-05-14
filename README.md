# Austin-animal-shelter
Analyzing cats and dogs data at the largest no-kill shelter in the United States

## Goals
1) Create a model to classify whether an animal at the shelter ends up with a "good" (e.g., adopted, returned to owner), "bad" (e.g., euthanized, lost), or "neutral" (e.g., transferred to a different shelter) outcome 
2) Use feature importance to find the features that best drive our classification
 
## Description of Data
Two csv files:

Austin_Animal_Center_Intakes: describes the animal at the time of "intake", i.e., when the animal enters the shelter. Columns include the time the animal enters, the breed, sex upon intake, the age of the animal, the color of the animal, where the animal was found, etc. Indexed by an 'Animal ID' unique to the animal.

Austin_Animal_Center_Outcomes: describes the animal at the time of "outcome", i.e., when the animal exited the shelter. Columns include the time the animal exits, the breed, sex upon outcome, the age of the animal, the color of the animal, what the outcome is (e.g., adopted or euthanized), etc. Indexed by an 'Animal ID' unique to the animal.

## Data Source
Austin Animal Shelter at the City of Austin Open Data Portal: https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Outcomes/9t4d-g238

An older version of this dataset can also be found at: https://www.kaggle.com/datasets/aaronschlegel/austin-animal-center-shelter-intakes-and-outcomes

## Notable Packages Used
1) scikit-learn: RandomForestClassifier
2) imblearn: SMOTENC and RandomUnderSampler to handle data imbalances
3) XGBoost: gradient boosted trees classifier

## Notebooks
We split our work to two notebooks:
1) Data cleaning and feature engineering (data_cleaning_and_feature_engineering.ipynb)
2) Modeling (modeling.ipynb)

## Data cleaning

### Merging the two data tables
The most important task in our data cleaning section is to merge the two data tables (Austin_Animal_Center_Intakes describing animal intakes and Austin_Animal_Center_Outcomes describing animal outcomes) into one dataframe with all the information that we require for our analysis. The issue is that there are animals who are repeat visitors to the shelter, so we cannot perform a merge on "Animal ID". To handle this, I created a new column that can be merged upon. This new column holds unique "visit IDs" combining the animal ID and the number of visitations, e.g., an animal with Animal ID A521520 that visits three times has three rows in the dataframes, each will have different visit IDs: A5215200, A5215201, and A5215202.

After merging, we ended up with one dataframe with the following features:
- visit_id           : unique ID for each animal visit to the shelter
- id                 : unique ID labeling each animal
- intake_time        : time the animal enters the shelter
- intake_MonthYear   : month/year the animal enters the shelter
- intake_loc         : where the animal was found
- intake_type        : how the animal was found (e.g., owner surrender, stray)
- intake_condition   : animal condition when it enters the shelter (e.g., sick)
- animal_type        : type of animal (e.g., cat, dog, bird)
- intake_sex         : sex of animal upon intake (e.g., spayed male)
- intake_age         : age of animal upon intake
- outcome_time       : time the animal exits the shelter (including via euthanasia)
- outcome_MonthYear  : month/year the animal exits the shelter
- outcome_type       : how the animal exits the shelter (e.g., adoption, return to owner, euthanasia)
- outcome_subtype    : extra information on how the animal exits the shelter (e.g., rabies risk)
- outcome_sex        : sex of animal upon outcome (e.g., spayed male)
- outcome_age        : age of animal upon outcome
- breed              : breed of animal
- color              : color of animal
- dob                : date of birth of the animal
- repeat             : whether the animal is a repeat visitor at the shelter
- mult               : how many times the animal has visited the shelter
 
### Dropping and imputing missing data
1) There is a lot of null values in outcome_subtype. This column contains miscellaneous information such as whether the animal is a 'Rabies Risk' or is currently 'At Vet'. While this column contains interesting information, I will not explore it in this project and thus dropped this column.
2) There are rows with missing income_sex, outcome_sex, income_age, outcome_type, intake_condition, breed, and outcome_age. I dropped rows with missing values on these columns except for outcome_age -- they represent a small percentage of the total data, so their removal should not affect the conclusion of the analysis. 
3) There are a few rows with missing outcome_age. This was filled by taking intake_age and adding to it the length of stay in the shelter, stay_length.
4) Duplicated information are deleted from the table (e.g., outcome_age, intake_age, and stay_length are collinear variables, we only need two of them in our data table).
5) Clearly anomalous data points (e.g., stay_length of less than 0 days) are dropped from the table. 
6) We want to focus on cats and dogs for this project, while this data table contains a small amount of other animals (e.g., birds, livestock). Non-cats/dogs rows are dropped from the table.
7) Breeds and colors with only a few entries per class are consolidated into a new class called 'Others'.

## Feature engineering
In the previous steps, I already engineered some new features that might be useful for our modeling:
1) 'stay_length_days': the length of stay in the shelter 
2) 'mult': the number of times the same animal visits the shelter

In this section, I engineered a two other features:

1) intake_day and intake_month: People often get pets at certain months of the year (e.g., Christmas) or at certain days of the week (e.g., weekends), so we can hypothize that the month or day of the week of outcome_time would be important predictors of an animal's outcome. However, outcome_time is not a good variable for predicting whether an animal will get adopted, as the outcome_time is recorded AFTER an animal exits the shelter. Instead, we want to know the chances for an animal to be adopted when they are still in the shelter! If we assume that there is some correlation between outcome_time and intake_time, we can instead hypothize that the month or day of the week of intake_time would be important predictors of an animal's outcome. Therefore, I generated the intake_day and intake_month features out of intake_time.
2) "good", "bad" or "neutral" outcomes: instead of trying to classify the outcome_type exactly, let us instead try to predict whether a particular animal's visit ended up with a good, bad, or neutral outcome. To do this, let us group the rows with good outcomes (Return to Owner, Adoption, or Rto-Adopt), bad outcomes (Euthanasia, Died, Missing, Disposal), and neutral outcomes (Transfer, Relocate).

### Best models
We performed a search over SARIMA order parameters and found that the best models to describe the time series are:

-) Export values: SARIMA(0,1,1)(2,0,0)[12]      
-) Export weights: SARIMA(0,1,0)(2,0,0)[12]   
-) Import values: SARIMA(0,1,1)(0,0,1)[12]      
-) Import weights: SARIMA(0,1,1)(0,0,2)[12]  

### Forecasts
![forecastexp_dol](https://user-images.githubusercontent.com/5288149/226216780-dd8a5f7c-1610-4f44-a4f6-a9033262abef.png)
![forecastexp_weight](https://user-images.githubusercontent.com/5288149/226216789-2aced4a9-8434-46db-99e5-b5febe4b9bee.png)
![forecastimp_dol](https://user-images.githubusercontent.com/5288149/226216794-260c6a01-a983-435a-9f8d-bd714421d886.png)
![forecastimp_weight](https://user-images.githubusercontent.com/5288149/226216796-8e10d996-791f-457f-9d1f-09f907b2b162.png)

The SMAPE values for our predictions are:

-) Export values: 4.92   
-) Export weights: 6.93   
-) Import values: 5.97      
-) Import weights: 6.39

The SMAPE values indicate a good fit, i.e., that we captured the general trend of the time series. However, the actual confidence intervals are quite wide -- certainly, too wide for me to be comfortable in using them to build a trading strategy or inform a political decision! This simply indicates the fact that the patterns in our time series is hard to predict. 

### Existence of cointegration
The result for the augmented Engle-Granger cointegration p-value between export value and import value is 0.056 while the p-value between export value and import weight is 0.076 (the null hypothesis is no cointegration). Neither is strong enough to claim existence of cointegration with significant confidence. If it is true that a healthy economy should strive for cointegration between exports and imports, this result argues that it might be necessary for Indonesia look into its export/import policies! 

## Possible improvements and future directions
We identified a few weaknesses in our analysis that could be improved in the future:

1) Looking at the residual QQ-plots and histograms, we found deviations from normality. This will affect the widths of the confidence intervals of our forecasts. Future work can perform modelling with other distributions (e.g., Student's t or skewed-t distributions).

2) Extending the time series included in the analysis from 2014 down to 1998 would give more baseline for the model to learn the pattern in the time series (especially the cointegration pattern). Care should be taken when using datapoints near 1998, as post-New Order policies might have not taken effect yet. 

Here are some future research directions that could be interesting to explore:

1) Performing similar analysis to particular sectors of imports/exports, e.g., just agricultural products, or even single items like coffee or cocoa.
2) Exploring relationships between Indonesia's imports/exports and various environmental time series (e.g., temperature or rainfall time series) with vector auto-regression.
3) Analyzing differences in the imports/exports time series between the post-New Order era and the New Order era.
