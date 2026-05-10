# Student Performance Analysis

This project is about student performance. I used Python to study the dataset, make graphs, do some statistical tests, and build a simple regression model to predict the final score.

The project has two main files:

- `student_performance_analysis.py`: this file runs the full analysis step by step.
- `app.py`: this file opens a Streamlit dashboard to explore the data.

If the file `student_performance.xlsx` is missing, the project creates sample data automatically, so the code can still run.

---

## Table of Contents

1. [Project Aim](#project-aim)
2. [Dataset Description](#dataset-description)
3. [Tools Used](#tools-used)
4. [Project Files](#project-files)
5. [Phase 1: Dataset Understanding](#phase-1-dataset-understanding)
6. [Phase 2: Exploratory Data Analysis](#phase-2-exploratory-data-analysis)
7. [Phase 3: Statistical Analysis](#phase-3-statistical-analysis)
8. [Phase 4: Outlier Detection](#phase-4-outlier-detection)
9. [Phase 5: Data Cleaning and Preprocessing](#phase-5-data-cleaning-and-preprocessing)
10. [Phase 6: Linear Regression Model](#phase-6-linear-regression-model)
11. [Phase 7: Final Conclusions](#phase-7-final-conclusions)
12. [Dashboard](#dashboard)
13. [How to Run](#how-to-run)
14. [Project Structure](#project-structure)
15. [Limitations](#limitations)

---

## Project Aim

The aim of this project is to understand what can affect the final score of students.

In this project, I tried to answer these questions:

1. Which columns are most related to `final_score`?
2. Do students with more study hours get better scores?
3. Does attendance help students get better results?
4. Can a regression model predict the final score?
5. Can the results be shown in a dashboard?

---

## Dataset Description

The dataset contains information about students. In the sample data, there are 500 students and 11 columns.

| Column | Type | Meaning |
|---|---|---|
| `student_id` | numerical | Student number |
| `age` | numerical | Student age |
| `gender` | categorical | Male or Female |
| `study_hours_per_week` | numerical | How many hours the student studies per week |
| `attendance_rate` | numerical | Attendance percentage |
| `parent_education` | categorical/ordinal | Education level of the parent |
| `internet_access` | categorical | If the student has internet access |
| `extracurricular` | categorical | If the student joins extra activities |
| `previous_score` | numerical | Previous exam score |
| `final_score` | numerical | Final exam score |
| `passed` | categorical | Yes or No |

The most important column is `final_score`, because this is the value that I want to explain and predict.

---

## Tools Used

| Tool | Why I Used It |
|---|---|
| Python | To write the code |
| pandas | To read and clean the data |
| numpy | To work with numbers |
| matplotlib | To make graphs |
| seaborn | To make better statistical graphs |
| scipy | To do statistical tests |
| statsmodels | To do the z-test |
| scikit-learn | To build the regression model |
| Streamlit | To create the dashboard |

---

## Project Files

| File or Folder | Use |
|---|---|
| `student_performance_analysis.py` | Full analysis script |
| `app.py` | Streamlit dashboard |
| `requirements.txt` | Required libraries |
| `README.md` | Project explanation |
| `plots/` | Saved graphs |

---

## Phase 1: Dataset Understanding

### What I Did

First, I loaded the dataset and checked the basic information:

- Number of rows and columns
- Column names
- Data types
- Missing values
- Duplicate rows
- Basic statistics like mean, median, minimum, and maximum

### Data Types and Possible Operations

Not all columns can be used in the same way. Numerical columns can use mathematical operations, but categorical columns are mostly used for counting and grouping.

| Data Type | Columns in This Project | Possible Operations |
|---|---|---|
| Numerical discrete | `student_id`, `age` | Count, minimum, maximum, grouping, charts |
| Numerical continuous | `study_hours_per_week`, `attendance_rate`, `previous_score`, `final_score` | Mean, median, standard deviation, correlation, regression, histograms, boxplots |
| Categorical nominal | `gender`, `internet_access`, `extracurricular`, `passed` | Count values, percentages, group comparison, bar charts |
| Categorical ordinal | `parent_education` | Order levels, count values, compare groups, encode into numbers |
| Target variable | `final_score` | Prediction, comparison, model error calculation |

### Basic Operators

These operations are mainly used with numerical columns:

| Operator | Meaning | Example |
|---|---|---|
| `+` | Addition | `final_score + previous_score` |
| `-` | Subtraction | `final_score - previous_score` |
| `*` | Multiplication | `study_hours_per_week * attendance_rate` |
| `/` | Division | `final_score / study_hours_per_week` |
| `%` | Modulo or percentage use | `attendance_rate / 100` |
| `==` | Equal to | `gender == "Female"` |
| `>` and `<` | Greater than / less than | `final_score > 60` |

In Python, division is written with `/`. The symbol `\` is not division. It is usually used in Windows paths or to continue a long line of code.

For example, it is useful to calculate the average of `final_score`, but it is not useful to calculate the average of `gender`. For `gender`, it is better to count how many males and females there are.

### Main Results

Using the sample data:

- The dataset has 500 rows and 11 columns.
- There are no duplicate rows.
- The average final score is about 66.63.
- The final score goes from about 38.8 to 96.4.
- The main numerical columns are age, study hours, attendance rate, previous score, and final score.

### Interpretation

The dataset is good for analysis because it has clear columns and enough rows. It has both numbers and categories, so I can use different methods.

### Phase Conclusion

The dataset is ready for the next step. It has useful information to study student performance.

---

## Phase 2: Exploratory Data Analysis

### What I Did

In this phase, I used graphs to understand the data better:

- Histograms
- Boxplots
- Countplots
- Correlation heatmap
- Pairplots
- Scatterplots

### Main Results

Important correlations with `final_score`:

| Variable | Correlation |
|---|---:|
| `study_hours_per_week` | 0.755 |
| `previous_score` | 0.224 |
| `attendance_rate` | 0.214 |
| `parent_education` | 0.060 |
| `age` | -0.041 |

### Interpretation

The strongest relationship is between study hours and final score. This means students who study more usually get better final scores.

Previous score and attendance also have a positive relationship, but it is weaker. Age, gender, and other categorical variables do not show a strong relationship with final score.

### Phase Conclusion

Study hours seem to be the most important variable. This is the main thing I noticed from the graphs.

---

## Phase 3: Statistical Analysis

### What I Did

I used statistical tests to check some ideas:

- Z-test
- T-test
- Pearson correlation
- Chi-square test
- ANOVA
- One-sample t-test

### Main Results

| Test | Result |
|---|---|
| Final score vs 50 | Significant |
| Gender and final score | Not significant |
| Internet access and final score | Significant in the sample data |
| Study hours and final score | Strong positive relationship |
| Attendance and final score | Weak positive relationship |
| Previous score and final score | Weak positive relationship |
| Gender and passed | Not significant |
| Parent education and final score | Not significant |

### Interpretation

The tests show that study hours are strongly linked with final score. Attendance and previous score also help, but not as much.

Gender does not seem to have an important effect. Parent education also does not show a strong difference in this sample.

### Phase Conclusion

The most useful variables are study hours, attendance, and previous score. Group variables are less important here.

---

## Phase 4: Outlier Detection

### What I Did

I checked outliers using:

- IQR method
- Z-score method

I checked these columns:

- `age`
- `study_hours_per_week`
- `attendance_rate`
- `previous_score`
- `final_score`

### Main Results

In the sample data:

- `age` and `attendance_rate` do not have important outliers.
- `study_hours_per_week` has a few possible outliers.
- `previous_score` has a few possible outliers.
- `final_score` has one possible outlier.

### Interpretation

The outliers are not too strange. They can still be real students. For example, one student may study much more than others, or one student may have a very low score.

### Phase Conclusion

The data is clean enough to continue. I did not need to remove many values.

---

## Phase 5: Data Cleaning and Preprocessing

### What I Did

Before using regression, I prepared the data:

- Filled missing `parent_education` values with the most common value.
- Changed categorical values into numbers.
- Encoded `parent_education` as ordered numbers.
- Used one-hot encoding for `gender`, `internet_access`, and `extracurricular`.
- Split the data into training and test sets.
- Scaled the features with `StandardScaler`.

### Features Used in the Model

The model uses:

- `age`
- `study_hours_per_week`
- `attendance_rate`
- `parent_education`
- `previous_score`
- `gender_Male`
- `internet_access_Yes`
- `extracurricular_Yes`

### Interpretation

The model needs numbers, so text values like `Male`, `Female`, `Yes`, and `No` must be converted. Scaling is also useful because the columns do not have the same unit.

### Phase Conclusion

After preprocessing, the data is ready for regression.

---

## Phase 6: Linear Regression Model

### What I Did

I trained a linear regression model to predict `final_score`.

I used these metrics:

- MAE: average error
- MSE: squared error
- RMSE: error in score points
- R2: how much the model explains the final score

### Model Results

Using the sample data:

| Metric | Train | Test |
|---|---:|---:|
| MAE | 4.72 | 4.92 |
| MSE | 35.36 | 42.37 |
| RMSE | 5.95 | 6.51 |
| R2 | 0.699 | 0.597 |

### Important Coefficients

| Feature | Coefficient |
|---|---:|
| `study_hours_per_week` | 8.17 |
| `attendance_rate` | 2.77 |
| `previous_score` | 2.40 |
| `parent_education` | 2.23 |
| `internet_access_Yes` | -0.31 |

### Interpretation

The model makes predictions with an average error of about 4.9 points on the test data. This is acceptable for a simple model.

The R2 score on the test data is about 0.597. This means the model explains about 59.7% of the final score variation.

Study hours are the strongest feature in the model, so this confirms what I saw in the EDA phase.

### Phase Conclusion

The regression model works and gives useful results. It is simple, but it helps understand which factors are important.

---

## Phase 7: Final Conclusions

### General Interpretation

The analysis shows that study habits are very important. Students who study more hours usually get better scores. Attendance and previous score also help, but less than study hours.

The background variables, like gender and parent education, are not very strong in this sample.

### Main Findings

1. Study hours are the strongest predictor of final score.
2. Attendance has a positive effect.
3. Previous score is useful but not the strongest feature.
4. Gender does not show a strong effect.
5. Parent education does not show a strong effect in this sample.
6. The regression model predicts with about 4.9 points average error.

### Final Conclusion

The main conclusion is that students should study regularly and attend classes. Schools can also help students who had low previous scores. The model is useful, but it should be used as support, not as the only way to judge students.

---

## Dashboard

The dashboard makes the project easier to explore. It has:

- Dataset overview
- Graphs
- Statistical tests
- Outlier detection
- Regression results
- Prediction form
- Summary page

The user can enter student information and predict the final score.

---

## How to Run

### 1. Install Libraries

```bash
pip install -r requirements.txt
```

### 2. Run the Analysis Script

```bash
python student_performance_analysis.py
```

This runs all phases and saves graphs in the `plots/` folder.

### 3. Run the Dashboard

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## Project Structure

```text
student-performance-analysis/
|
|-- app.py
|-- student_performance_analysis.py
|-- requirements.txt
|-- README.md
|-- plots/
|   |-- histograms.png
|   |-- boxplots.png
|   |-- countplots.png
|   |-- correlation_heatmap.png
|   |-- pairplot.png
|   |-- scatterplots.png
|   |-- regression_results.png
```

---

## Limitations

This project has some limits:

- If the Excel file is missing, the project uses generated sample data.
- Sample data may not be exactly like real student data.
- Linear regression only models straight-line relationships.
- Other important factors are missing, like motivation, family income, health, and teacher support.
- The prediction should help decision-making, but it should not replace human judgment.

---

## Short Summary

In this project, I analyzed student performance using Python. I found that study hours are the most important factor for the final score. I also built a simple regression model and a dashboard to show the results.
