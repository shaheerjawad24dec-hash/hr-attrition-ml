# Kaggle dataset

**IBM HR Analytics Employee Attrition & Performance**
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

- File: `WA_Fn-UseC_-HR-Employee-Attrition.csv` (1,470 rows x 35 columns, no missing values)
- Fictional data created by IBM data scientists for attrition analysis.
- A copy is already in `data/`, so no Kaggle account is needed to run the project.

To download it yourself: sign in to Kaggle, open the link, click **Download**,
unzip, and place `WA_Fn-UseC_-HR-Employee-Attrition.csv` in the `data/` folder.

## Columns used

| Column | Meaning | Used by |
| --- | --- | --- |
| Attrition | Did the employee leave? (Yes/No) | Classification **target** |
| MonthlyIncome | Monthly pay in USD | Regression **target**; attrition feature |
| JobRole | One of 9 roles | Both |
| JobLevel | 1 (junior) to 5 (executive) | Both |
| TotalWorkingYears | Career length in years | Both |
| YearsAtCompany | Tenure at this company | Both |
| YearsInCurrentRole | Years in current role | Salary |
| Education | 1 Below College, 2 College, 3 Bachelor, 4 Master, 5 Doctor | Salary |
| YearsWithCurrManager | Years under current manager | Attrition |
| NumCompaniesWorked | Previous employers | Attrition |
| OverTime | Works overtime (Yes/No) | Attrition |
| BusinessTravel | Non-Travel, Travel_Rarely, Travel_Frequently | Attrition |
| DistanceFromHome | Commute distance (1-29) | Attrition |
| StockOptionLevel | 0-3 | Attrition |
| JobSatisfaction, EnvironmentSatisfaction, JobInvolvement | 1 Low, 2 Medium, 3 High, 4 Very High | Attrition |
| WorkLifeBalance | 1 Bad, 2 Good, 3 Better, 4 Best | Attrition |

Not used: `Gender`, `Age`, `MaritalStatus` (fairness), `EmployeeCount`,
`Over18`, `StandardHours` (constant), `EmployeeNumber` (ID), and the remaining
columns (little added value in cross-validation).
