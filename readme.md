# Zestimate Error Drivers Project


1. Here is a <a href="https://trello.com/b/EMEzPn69/clusteringproject">link</a> to my Trello board I used for project planning.

2. Goal - Continuing with the Zillow 2017 properties and predictions for single unit family homes I am looking for what is driving the errors in the Zestimates.

## Project Details
- Libraries
    - pandas
    - numpy
    - datetime
    - sklearn
    - seaborn
    - matplotlib
    - scipy
- Individual modules
    - wrangle
    - explore
    - evaluate
    - model
    - env

## Pipeline
My methodology follow is the data pipeline; plan, acquire, prepare, explore, model and deliver.
### Plan
- My plan is to take the 2017 properties and predictions through the data pipeline in order to find log error drivers and prepare a model to predict future errors.
### Acquire
- I acquired my data via the zillow SQL database, importing all tables (LEFT JOIN), sub query DISTINCT id from 2017 properties, WHERE statement for 2017% from the predictions table and where latitude iS NOT NULL.  
### Prepare
- Drop or filled in rows and columns based on null count.
- Single unit family homes.
- Renamed columns.
- Calculated age, tax_rate and price_per_sqft.
- COnverted ints.
### Explore
- Histograms
- Regplo
- Heatmaps
- Clusters based on exploration of unscaled data.
    - longitude & Latitude
    - logerror & age
    - logerror & sqft
    - logerror & lot_size
    - logerror & home values
### Model
- Feature engineering using rfe and select_kbest
- Regression modeling
    - Model performed better then baseline
- Linear Regression - Best model (test)
- LassoLars
- Tpolynomial Regression
## Hypothesis
- Is a higher log error dependent on homes over 50 years old? (Cluster - 2) - Rejected the null, these are dependent of each other.
- Is a higher log error dependent on homes less 1000 sqft? (Cluster - 6) - Rejected the null, these are dependent of each other.
- Is a higher log error dependent on homes who's ppsqft is less 200? (Cluster - 7) - Failed to reject the null, they are independent.
- Is a higher log error dependent on homes with a smaller lot size? (Cluster - 8) - Failed to reject the null, they are independent.
- Is a higher log error dependent on less expensive homes? (Cluster - 9) - Rejected the null, these are dependent of each other.
## Key findings and takeaways
- Majority of homes have 2-4 bathrooms.
- Majority of homes are 2-4 bedrooms.
- Majoprity of homes are < 2,500 sqft.
- Homes are in three different counties with the majority of homes in 6040.
- Majority of homes are < 75 years old.
##### Stats Testing
- Homes older than 50 years old have a higher log error rate.
- Homes less than 1000sqft have a higher log error rate.
- Homes less than 250,000 have a higher log error rate.
##### Feature engineering
- What is interesting here is age does not factor into any of the tests.
- Sqft is the best feature.
##### Modeling
- Modeling on sqft and log error the model did better then the baseline.
- Train RMSE on the Liner Regression model preformed better then the other two models.
- With an RMSE of 0.1657 I chose to test on the liners regression model.
##### Test Model
- Test model did not perform as well as I hope going over by .100th of a point with an RMSE of 0.175 using sqft and logerror.
# Data Dictionary
| Value          | Description                                                           | DataType |
|----------------|-----------------------------------------------------------------------|----------|
| bathrooms      | bathroom count.                                                       | int64    |
| bedrooms       | bedroom count.                                                        | int64    |
| sqft           | square feet of home.                                                  | int64    |
| county_code    | code identifying the county the home is in.                           | int64    |
| latitude       | maps latitude                                                         | float64  |
| longitude      | maps longitude                                                        | float64  |
| lot_size       | size of the lot the home sits on.                                     | int64    |
| tax_value      | value of the home.                                                    | int64    |
| logerror       | Zestimate error rate.                                                 | float64  |
| county         | county name based on county code.                                     | object   |
| age            | age of the home calculated from today's date and year built.          | int64    |
| tax_rate       | tax rate paid on the property calculated by tax amount and tax value. | float64  |
| price_per_sqft | price per square foot calculated by sqft and tax value.               | int64    |
| abs_logerror   | absolute value of log error                                           | float64  |

