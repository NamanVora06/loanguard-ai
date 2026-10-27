# Dataset Documentation: Default of Credit Card Clients

## 1. Source and Reference
- **Dataset Name:** Default of Credit Card Clients Dataset
- **Original Source:** UCI Machine Learning Repository
- **Creators:** I-Cheng Yeh (Tamkang University) and Che-hui Lien (Chung Hua University)
- **Reference Paper:** Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473-2480.
- **License:** Open Database License (ODbL) / Public Domain per UCI terms for educational and research purposes.

## 2. Target Definition
- **Target Variable:** `default_payment_next_month` (`1` = Default, `0` = Non-Default)
- **Meaning:** Indicates whether the customer failed to pay their minimum required credit card balance in the subsequent month (October 2005).
- **Target Imbalance:** Baseline default rate is approximately **22.1%** (~6,636 defaults out of 30,000 clients).

## 3. Feature Descriptions
The dataset contains 23 input features:

### Demographics
- `LIMIT_BAL`: Amount of given credit in NT dollars (individual consumer credit and supplementary family credit).
- `SEX`: Gender (1 = Male, 2 = Female).
- `EDUCATION`: Education level (1 = Graduate School, 2 = University, 3 = High School, 4 = Others).
- `MARRIAGE`: Marital status (1 = Married, 2 = Single, 3 = Others).
- `AGE`: Age in years (21–75).

### Repayment History (Past 6 Months: April – September 2005)
- `PAY_0`: Repayment status in September 2005.
- `PAY_2`: Repayment status in August 2005.
- `PAY_3`: Repayment status in July 2005.
- `PAY_4`: Repayment status in June 2005.
- `PAY_5`: Repayment status in May 2005.
- `PAY_6`: Repayment status in April 2005.
  - Scale: `-2` = No consumption/inactive, `-1` = Paid in full, `0` = Revolving credit (minimum paid), `1` = Payment delay for 1 month, `2` = Delay for 2 months, ..., `8` = Delay for 8+ months.

### Bill Statement Amounts (NT$)
- `BILL_AMT1` to `BILL_AMT6`: Statement balance from September back to April 2005.

### Previous Payment Amounts (NT$)
- `PAY_AMT1` to `PAY_AMT6`: Actual amount paid from September back to April 2005.

## 4. Prevention of Data Leakage
- `ID` column has been excluded as an arbitrary identifier.
- Only historical payment, billing, and demographic records available *prior* to the loan period are used as predictors.
- No future payment data or post-decision variables are included.
- Preprocessing imputations and scalers are fitted exclusively on the training split (70%) and evaluated on the held-out test split (15%).

## 5. Limitations & Ethical Notice
- **Educational Prototype:** This system is an academic machine learning prototype and must **not** be used for real-world automated credit underwriting or lending decisions.
- **Historical and Geographic Context:** Data originates from Taiwanese credit card holders in 2005 and may not generalize across different macroeconomic conditions or modern credit products without retraining.
- **Demographic Features:** Standard lending regulations (such as the US Equal Credit Opportunity Act) prohibit the use of protected classes like gender and marital status in credit decisions. In this project, demographic fields are included strictly to reproduce the original benchmark study.
