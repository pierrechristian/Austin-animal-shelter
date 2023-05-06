# Austin-animal-shelter
Analyzing animal shelter data at the largest no-kill shelter in the United States

## Goals
1) Create a model to forecast the Indonesian non-oil/gas import and export monthly time series the year after the end of the available dataset
2) Test for the existence of cointegration on the import and export time series

## Context

### Indonesia's role in the commodity sector
Indonesia is one of the world's largest exporter of raw and agricultural products. Its main exports (besides oil and gas products) include minerals, electrical appliances, and various agricultural goods. It is the world's largest supplier of cloves, and a major supplier of crude palm oil, coffee, and rubber. 

### Cointegration of import and export 
Policies tend to create convergence between imports and exports. This has been tested for 50 countries, where over the quarterly period of 1973 to 1998, where all 50 countries (except for Mexico) show strong signs of cointegration between imports and exports (https://www.sciencedirect.com/science/article/abs/pii/S1059056001001010). 

However, 1998 is the end of the New Order Era in Indonesia. Post 1998, the Indonesian government placed many new trade policies that might changed the long-run relationships between import and export. I would like to check on whether cointegration between import and export is maintained about a decade after the end of the New Order Era.

## Description of Data
We have four time series recording the monthly export values (in dollars), export weights (in kg), import values (in dollars), and import values (in kg) ranging from January 2014 to December 2022. 

## Data Source
Indonesia' Badan Pusat Statistik (Central Statistical Body): https://www.bps.go.id/subject/8/ekspor-impor.html#subjekViewTab3

## Technologies Used
1) pmdarima: SARIMA to model and forecast the time series 
2) statsmodels: Augmented Engle-Granger two-step cointegration test and Ljung-Box test for time-series residuals

## Results

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
