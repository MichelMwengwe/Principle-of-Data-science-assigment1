# Question 1 — Frailty Analysis

## Dataset

The dataset contains information from 10 female participants. The variables include height, weight, age, grip strength, and frailty status.

- Height is measured in inches.
- Weight is measured in pounds.
- Age is measured in years.
- Grip strength is measured in kilograms.
- Frailty is recorded as Y for frail and N for not frail.

## Data Processing

The following processing steps were completed:

### Unit Standardization
1. Height was converted from inches to meters.
2. Weight was converted from pounds to kilograms.
### Feature Engineering
1. Body Mass Index (BMI) was calculated using weight in kilograms divided by height in meters squared.
2. Participants were placed into age groups:
   - `<30`
   - `30–45`
   - `46–60`
   - `>60`
### Categorical-to-Numeric Encoding
1. Frailty was converted from text to a binary variable:
   - Y = 1
   - N = 0
2. Age groups were one-hot encoded into separate numeric columns.
### EDA and Reporting
The processed data was saved as `frailty_processed.csv`. The script also calculated the mean, median, and standard deviation for the numeric variables. It additionally summarized the binary frailty variable and the one-hot encoded age-group columns.

## Summary Results

There were 4 frail participants and 6 participants classified as not frail.

|    Variable         | Mean  | Median | Standard Deviation |
|---------------------|-------|------:|--------------------:|
| Height in inches    | 68.60 | 68.45 |      1.67           |
| Weight in pounds    | 131.90| 136.00|      14.23          |
|        Age          | 32.50 | 29.50 |      12.86          |
|  Grip strength      | 26.00 | 27.00 |      4.52           |
| Height in meters    | 1.742 | 1.739 |      0.042          |
| Weight in kilograms | 59.83 | 61.69 |      6.46           |
| BMI                 | 19.682| 19.185|      1.781          |

## Grip Strength and Frailty

The Pearson correlation between grip strength and the binary frailty variable was: `-0.4759`
The negative correlation indicates that, in this sample, higher grip strength tends to be associated with a lower frailty indicator. However, the dataset contains only 10 participants, so this result describes an association and does not prove that grip strength causes or prevents frailty.