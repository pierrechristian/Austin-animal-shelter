# Austin-animal-shelter
Analyzing cats and dogs data at the largest no-kill shelter in the United States.

## Goals
1) Create a model to classify whether an animal at the shelter ends up with a "good" (e.g., adopted, returned to owner), "bad" (e.g., euthanized, lost), or "neutral" (e.g., transferred to a different shelter) outcome; a key challenge in accomplishing this goal is that the dataset is heavily **imbalanced** (very few "bad" outcomes)
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

## Modeling
In this section, I want to predict whether an animal's outcome is good, bad, or neutral. A key point in this analysis is the imbalanced dataset; because this animal shelter is "no-kill", there will be very few euthanasia. In fact, most outcomes for the animals are good (e.g., adopted or returned to owner), and there are very few bad (e.g., euthanasia) outcomes! 

![imbalance_dataset](https://github.com/pierrechristian/Austin-animal-shelter/assets/5288149/fedfa773-1bff-4cb3-b498-2f62b094e861)

We will use two models, **random forest** and **gradient boosted trees**. We will also try multiple methods for dealing with the imbalanced dataset!

### Random Forest
My strategy with applying random forest on this imbalanced dataset is the following:

1) Rebalance or weight the data
2) Perform grid-search cross-validation to find the optimal random forest parameters
3) Compare the weighted f1 score on the holdout set (20%) between the different rebalance/data-weighting schemes to find the best model

We will use three rebalancing/data-weighting schemes to handle the imbalanced dataset:

1) SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and Continuous): this is an oversampling algorithm that creates synthetic data for the minority group, and is an extension of SMOTE that allows for categorical features. 
2) random undersample: undersample the majority groups by taking random samples of it. This technique is much faster than SMOTENC, but come at the cost of greatly reducing our sample size.
3) class-weights: the random forest algorithm allows for a natural way to weigh different classes, so we can use class weights on the random forest algorithm to put different emphases on minority/majority groups instead of resampling the training set.

The resulting weighted f1-scores on the holdout set are the following:
1) SMOTENC: 0.82
2) random undersample: 0.79
3) class-weights: 0.34

Comparing the three random forest models (which differs based on how we handle the data imbalances), the one with the best weighted f1-score is class-weights with a weighted f1 of 0.83. This is very slightly greater than the weighted f1-score of SMOTENC (0.82) and markedly better than the weighted f1-score of random undersample (0.79).

However, the f1-score of the "bad" outcome for class-weights (0.34) is significantly worse than SMOTENC (0.39) and random undersample (0.42), owing to class-weights having poor recall for "bad" oucomes (0.24). As identifying the bad outcomes correctly is important in this problem, I do not think that the slight increase of 0.01 point in weighted f1-score in using the class-weights instead of SMOTENC is worth it. Thus, I declare our best random forest model to be the SMOTENC model.

### Gradient Boosted Trees with XGBoost
Next, I also modeled the dataset with gradient boosted trees. At the end we can compare its performance on the holdout set (20%) with the random forest models to obtain our final model.

As with random forest, gradient boosted trees also lend itself to class weightings to handle the dataset imbalance. In this section I tried XGBoost both with and without class weights. As before, the optimal XGBoost parameters were found by performing grid-search cross-validation.

The resulting weighted f1-scores on the holdout set are the following:
1) Without class-weights: 0.80
2) With class-weights: 0.81

The weighted f1-score for the two XGBoost models are smaller than our best random forest model (SMOTENC), so **our best overall model is the random forest model with SMOTENC**.

### Best Model: Random Forest with SMOTENC

Here are the performance metrics of our best performing model on the holdout set, the random forest model with SMOTENC to rebalance the dataset. Just as a reminder our task is to predict each datapoint as having "bad", "good", or "neutral" outcome. The data is imbalanced, with few examples of "bad" outcomes.

| outcome | precision | recall | f1-score   |
| ------------- | ------------- | ------------- | ------------- |
| bad  |0.35  |    0.45    |  0.39    |
| good  | 0.87  |    0.91   |   0.89 |
|neutral | 0.74    |  0.63    |  0.68 |
|weighted average | 0.82    |  0.82    |  0.82 |

## Feature Importance

Now that we have a a best overall model, we can see which of the features drive the classification the most. We can do this by computing the permutation importance:

![permutation_importance](https://github.com/pierrechristian/Austin-animal-shelter/assets/5288149/18607bc8-7873-44e7-b47f-03053362b5a8)


The permutation importance plot reveals that the top three features that are most important are:

1) outcome_sex: in addition to male/female, the "outcome_sex" feature records whether the animal has been spayed/neutered at the time of outcome. There are two reasons of why this feature might be the one that is most important. First, animals that are already spayed/neutered are more attractive to adopters as spaying/neutering costs money to do. The second reason is that the "good" outcome include "return_to_owner", indicating that they are lost pets instead of wild animals, and most pet cats/dogs in the US are spayed/neutered. 

2) stay_length_days: this feature records the length of time the animal is in the shelter. A lost pet is quickly picked up by their owners, resulting in a good outcome within a short amount of time. 

3) animal_type: indicates cats vs dogs. It turns out that dogs are slightly (about 5% more) more favored to have good outcome than cats.

## Future Directions

1) Performing a similar analysis on a shelter that is not non-kill: particularly I am interested in possible differences in features that best decide an animal getting a bad outcome
2) Performing a similar analysis on a shelter from a different country: different cultures have different predilections on what makes a favorable pet (e.g., dogs vs cats, different attitudes on animal spaying/neutering)
3) Time series analysis: an investigation on how the feature importance plot changes across time can show how attitudes and practices on pet adoption and shelter care changes with time 
