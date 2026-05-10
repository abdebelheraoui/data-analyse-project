"""
============================================================================
STUDENT PERFORMANCE ANALYSIS - COMPLETE DATA SCIENCE PROJECT
============================================================================
A university-level data science project analyzing student performance data.
Includes: EDA, Statistical Analysis, Outlier Detection, Linear Regression,
and comprehensive visualizations.

Author: Data Science Student
Date: 2026
Dataset: Student Performance Dataset (500 students, 11 features)
============================================================================
"""

# ============================================================
# IMPORTS
# ============================================================
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.weightstats import ztest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
Path('plots').mkdir(exist_ok=True)

EDU_MAPPING = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}

def generate_sample_data(n=500, seed=42):
    """Create a reproducible dataset when no spreadsheet is available."""
    rng = np.random.default_rng(seed)
    study_hours = rng.normal(15, 5, n).clip(2, 30).round(1)
    attendance = rng.normal(76, 12, n).clip(50, 100).round(1)
    previous_score = rng.normal(65, 12, n).clip(30, 95).round(1)
    parent_education = rng.choice(
        list(EDU_MAPPING.keys()),
        n,
        p=[0.35, 0.35, 0.22, 0.08]
    )
    final_score = (
        8
        + study_hours * 1.65
        + attendance * 0.22
        + previous_score * 0.22
        + pd.Series(parent_education).map(EDU_MAPPING).to_numpy() * 1.5
        + rng.normal(0, 6, n)
    ).clip(0, 100).round(1)

    return pd.DataFrame({
        'student_id': np.arange(1, n + 1),
        'age': rng.integers(15, 20, n),
        'gender': rng.choice(['Male', 'Female'], n),
        'study_hours_per_week': study_hours,
        'attendance_rate': attendance,
        'parent_education': parent_education,
        'internet_access': rng.choice(['No', 'Yes'], n, p=[0.25, 0.75]),
        'extracurricular': rng.choice(['No', 'Yes'], n, p=[0.55, 0.45]),
        'previous_score': previous_score,
        'final_score': final_score,
        'passed': np.where(final_score >= 60, 'Yes', 'No')
    })

def load_student_data():
    """Load the project dataset, falling back to reproducible sample data."""
    data_path = Path('student_performance.xlsx')
    if data_path.exists():
        return pd.read_excel(data_path)

    csv_path = Path('student_performance.csv')
    if csv_path.exists():
        return pd.read_csv(csv_path)

    print("student_performance.xlsx not found; using generated sample data.")
    return generate_sample_data()

# ============================================================
# PHASE 1: DATASET UNDERSTANDING
# ============================================================
print("=" * 60)
print("PHASE 1: DATASET UNDERSTANDING")
print("=" * 60)

# Load dataset
df = load_student_data()

# Basic information
print(f"\nDataset Shape: {df.shape}")
print(f"Total Cells: {df.size}")
print(f"\nColumn Names: {df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")

# Missing values
missing = df.isnull().sum()
print(f"\nMissing Values:\n{missing[missing > 0]}")
print(f"Total Missing: {df.isnull().sum().sum()}")

# Duplicates
print(f"\nDuplicate Rows: {df.duplicated().sum()}")

# Descriptive statistics
print(f"\nDescriptive Statistics:\n{df.describe()}")

# Categorical columns analysis
categorical_cols = df.select_dtypes(include=['object']).columns
print("\nCategorical Columns Unique Values:")
for col in categorical_cols:
    print(f"\n{col}: {df[col].nunique()} unique values")
    if df[col].nunique() <= 20:
        print(df[col].value_counts())

# Numerical columns
numerical_cols = ['age', 'study_hours_per_week', 'attendance_rate', 'previous_score', 'final_score']
print("\nNumerical Columns Distribution Summary:")
for col in numerical_cols:
    print(f"\n{col}:")
    print(f"  Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}")
    print(f"  Std: {df[col].std():.2f}, Range: [{df[col].min()}, {df[col].max()}]")
    print(f"  Skewness: {df[col].skew():.3f}")

print("\n[OK] PHASE 1 COMPLETE")

# ============================================================
# PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

# 2.1 Histograms
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(numerical_cols):
    axes[i].hist(df[col], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[i].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    axes[i].axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean: {df[col].mean():.1f}')
    axes[i].legend()
axes[5].remove()
plt.suptitle('Histograms of Numerical Variables', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/histograms.png', dpi=150, bbox_inches='tight')
plt.show()

# 2.2 Boxplots
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, col in enumerate(numerical_cols):
    axes[i].boxplot(df[col], patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='navy'),
                    medianprops=dict(color='red', linewidth=2))
    axes[i].set_title(col, fontsize=12, fontweight='bold')
plt.suptitle('Boxplots of Numerical Variables', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/boxplots.png', dpi=150, bbox_inches='tight')
plt.show()

# 2.3 Countplots for categorical variables
cat_plot_cols = ['gender', 'parent_education', 'internet_access', 'extracurricular', 'passed']
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(cat_plot_cols):
    value_counts = df[col].value_counts()
    colors = sns.color_palette("husl", len(value_counts))
    bars = axes[i].bar(value_counts.index.astype(str), value_counts.values, color=colors, edgecolor='black')
    axes[i].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
    for bar in bars:
        height = bar.get_height()
        axes[i].text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)
axes[5].remove()
plt.suptitle('Countplots of Categorical Variables', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/countplots.png', dpi=150, bbox_inches='tight')
plt.show()

# 2.4 Correlation Heatmap
df_encoded = df.copy()
df_encoded['gender'] = df_encoded['gender'].map({'Male': 0, 'Female': 1})
df_encoded['internet_access'] = df_encoded['internet_access'].map({'No': 0, 'Yes': 1})
df_encoded['extracurricular'] = df_encoded['extracurricular'].map({'No': 0, 'Yes': 1})
df_encoded['passed'] = df_encoded['passed'].map({'No': 0, 'Yes': 1})
df_encoded['parent_education'] = df_encoded['parent_education'].map(EDU_MAPPING)
df_corr = df_encoded.drop('student_id', axis=1)

corr_matrix = df_corr.corr()
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, mask=mask)
plt.title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nCorrelation with final_score:")
print(corr_matrix['final_score'].sort_values(ascending=False))

# 2.5 Pairplot
pair_cols = ['study_hours_per_week', 'attendance_rate', 'previous_score', 'final_score']
pairplot = sns.pairplot(df[pair_cols], diag_kind='kde', plot_kws={'alpha': 0.6, 's': 50})
pairplot.fig.suptitle('Pairplot of Key Numerical Variables', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/pairplot.png', dpi=150, bbox_inches='tight')
plt.show()

# 2.6 Scatterplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.regplot(data=df, x='study_hours_per_week', y='final_score', ax=axes[0],
            scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
axes[0].set_title('Study Hours vs Final Score', fontweight='bold')
sns.regplot(data=df, x='attendance_rate', y='final_score', ax=axes[1],
            scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
axes[1].set_title('Attendance vs Final Score', fontweight='bold')
sns.regplot(data=df, x='previous_score', y='final_score', ax=axes[2],
            scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
axes[2].set_title('Previous Score vs Final Score', fontweight='bold')
plt.suptitle('Scatterplots: Key Predictors vs Final Score', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/scatterplots.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n[OK] PHASE 2 COMPLETE")

# ============================================================
# PHASE 3: STATISTICAL ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3: STATISTICAL ANALYSIS")
print("=" * 60)

# 3.1 Z-Test
print("\n--- Z-Test: Final Score vs 50 ---")
z_stat, z_pvalue = ztest(df['final_score'], value=50)
print(f"Z-statistic: {z_stat:.4f}, P-value: {z_pvalue:.6f}")
print(f"H0: mean = 50, H1: mean != 50")
print(f"Result: {'Reject H0' if z_pvalue < 0.05 else 'Fail to reject H0'}")

# 3.2 Independent T-Test: Gender
print("\n--- T-Test: Final Score by Gender ---")
male_scores = df[df['gender'] == 'Male']['final_score']
female_scores = df[df['gender'] == 'Female']['final_score']
t_stat, t_pvalue = stats.ttest_ind(male_scores, female_scores)
print(f"Male mean: {male_scores.mean():.2f}, Female mean: {female_scores.mean():.2f}")
print(f"T-statistic: {t_stat:.4f}, P-value: {t_pvalue:.6f}")
print(f"Result: {'Reject H0' if t_pvalue < 0.05 else 'Fail to reject H0'}")

# 3.3 T-Test: Internet Access
print("\n--- T-Test: Final Score by Internet Access ---")
inet_yes = df[df['internet_access'] == 'Yes']['final_score']
inet_no = df[df['internet_access'] == 'No']['final_score']
t_stat2, t_pvalue2 = stats.ttest_ind(inet_yes, inet_no)
print(f"T-statistic: {t_stat2:.4f}, P-value: {t_pvalue2:.6f}")

# 3.4 Pearson Correlation Tests
print("\n--- Pearson Correlation Tests ---")
for var in ['study_hours_per_week', 'attendance_rate', 'previous_score']:
    r, p = stats.pearsonr(df[var], df['final_score'])
    print(f"{var} vs final_score: r = {r:.4f}, p = {p:.6f}")

# 3.5 Chi-Square Test
print("\n--- Chi-Square: Gender vs Passed ---")
contingency = pd.crosstab(df['gender'], df['passed'])
chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency)
print(f"Chi-square: {chi2:.4f}, P-value: {chi2_p:.6f}")

# 3.6 ANOVA
print("\n--- ANOVA: Final Score by Parent Education ---")
df_anova = df.dropna(subset=['parent_education'])
groups = [g['final_score'].values for _, g in df_anova.groupby('parent_education')]
f_stat, anova_p = stats.f_oneway(*groups)
print(f"F-statistic: {f_stat:.4f}, P-value: {anova_p:.6f}")

# 3.7 One-Sample T-Test
print("\n--- One-Sample T-Test: Final Score vs 60 ---")
t_stat_60, t_pvalue_60 = stats.ttest_1samp(df['final_score'], 60)
print(f"T-statistic: {t_stat_60:.4f}, P-value: {t_pvalue_60:.6f}")

print("\n[OK] PHASE 3 COMPLETE")

# ============================================================
# PHASE 4: OUTLIER ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("PHASE 4: OUTLIER ANALYSIS")
print("=" * 60)

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    iqr_outliers = len(df[(df[col] < lower) | (df[col] > upper)])
    z_scores = np.abs(stats.zscore(df[col]))
    z_outliers = len(df[z_scores > 3])
    print(f"{col}: IQR outliers = {iqr_outliers}, Z-score outliers = {z_outliers}")

print("\n[OK] PHASE 4 COMPLETE")

# ============================================================
# PHASE 5: DATA CLEANING & PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("PHASE 5: DATA CLEANING & PREPROCESSING")
print("=" * 60)

# Handle missing values
df_clean = df.copy()
df_clean['parent_education'] = df_clean['parent_education'].fillna(df_clean['parent_education'].mode()[0])

# Encode categorical variables
df_encoded = pd.get_dummies(df_clean, columns=['gender', 'internet_access', 'extracurricular'], drop_first=True)
df_encoded['parent_education'] = df_encoded['parent_education'].map(EDU_MAPPING)
df_encoded['passed'] = df_encoded['passed'].map({'No': 0, 'Yes': 1})

# Feature selection
X = df_encoded.drop(columns=['student_id', 'passed', 'final_score'])
y = df_encoded['final_score']
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Features: {X.shape[1]}, Training: {X_train.shape[0]}, Test: {X_test.shape[0]}")
print("\n[OK] PHASE 5 COMPLETE")

# ============================================================
# PHASE 6: LINEAR REGRESSION MODEL
# ============================================================
print("\n" + "=" * 60)
print("PHASE 6: LINEAR REGRESSION MODEL")
print("=" * 60)

# Train model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = lr_model.predict(X_train_scaled)
y_test_pred = lr_model.predict(X_test_scaled)

# Evaluation metrics
metrics = {
    'MAE': [mean_absolute_error(y_train, y_train_pred), mean_absolute_error(y_test, y_test_pred)],
    'MSE': [mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)],
    'RMSE': [np.sqrt(mean_squared_error(y_train, y_train_pred)), np.sqrt(mean_squared_error(y_test, y_test_pred))],
    'R2': [r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)]
}

print("\n+--------------+----------+----------+")
print("| Metric       | Train    | Test     |")
print("+--------------+----------+----------+")
for metric, values in metrics.items():
    print(f"| {metric:12s} | {values[0]:8.4f} | {values[1]:8.4f} |")
print("+--------------+----------+----------+")

# Feature coefficients
print("\nFeature Coefficients:")
feature_names = X.columns.tolist()
for name, coef in zip(feature_names, lr_model.coef_):
    print(f"  {name}: {coef:.4f}")

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_train, y_train_pred, alpha=0.6, color='steelblue', edgecolors='black', linewidth=0.5)
min_val, max_val = min(y_train.min(), y_train_pred.min()), max(y_train.max(), y_train_pred.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
axes[0].set_title(f'Training Set - R² = {metrics["R2"][0]:.4f}', fontweight='bold')
axes[0].set_xlabel('Actual'); axes[0].set_ylabel('Predicted')

axes[1].scatter(y_test, y_test_pred, alpha=0.6, color='green', edgecolors='black', linewidth=0.5)
min_val, max_val = min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
axes[1].set_title(f'Test Set - R² = {metrics["R2"][1]:.4f}', fontweight='bold')
axes[1].set_xlabel('Actual'); axes[1].set_ylabel('Predicted')
plt.suptitle('Linear Regression: Actual vs Predicted', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/regression_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n[OK] PHASE 6 COMPLETE")

# ============================================================
# PHASE 7: FINAL CONCLUSIONS
# ============================================================
print("\n" + "=" * 60)
print("PHASE 7: FINAL CONCLUSIONS")
print("=" * 60)

conclusions = """
KEY FINDINGS:
1. Study hours per week is the strongest predictor (r = 0.804)
2. Attendance rate moderately correlates with final score (r = 0.241)
3. Previous score has weak correlation (r = 0.165)
4. Gender, internet access show NO significant effect on performance
5. Model explains 75.3% of variance (R² = 0.753)
6. Average prediction error: ~6.3 points (MAE)

RECOMMENDATIONS:
- Encourage students to increase study hours (strongest factor)
- Monitor and improve attendance rates
- Provide support for students with low previous scores
- Gender-neutral policies are statistically justified

LIMITATIONS:
- Synthetic dataset may not reflect real-world complexity
- 23.4% missing parent education data
- Linear model assumes linear relationships only
"""
study_corr = corr_matrix.loc['study_hours_per_week', 'final_score']
attendance_corr = corr_matrix.loc['attendance_rate', 'final_score']
previous_corr = corr_matrix.loc['previous_score', 'final_score']
test_r2 = metrics['R2'][1]
test_mae = metrics['MAE'][1]

conclusions = f"""
KEY FINDINGS:
1. Study hours per week is the strongest predictor (r = {study_corr:.3f})
2. Attendance rate correlation with final score: r = {attendance_corr:.3f}
3. Previous score correlation with final score: r = {previous_corr:.3f}
4. Gender and internet access are weaker predictors than study behavior in this model
5. Model explains {test_r2 * 100:.1f}% of variance on the test set (R2 = {test_r2:.3f})
6. Average test prediction error: ~{test_mae:.1f} points (MAE)

RECOMMENDATIONS:
- Encourage students to increase study hours (strongest factor)
- Monitor and improve attendance rates
- Provide support for students with low previous scores
- Use the regression as a screening aid, not as the only decision source

LIMITATIONS:
- Generated fallback data may not reflect real-world complexity when no spreadsheet is present
- Linear model assumes linear relationships only
"""
print(conclusions)

print("=" * 60)
print("PROJECT COMPLETE - ALL PHASES FINISHED SUCCESSFULLY")
print("=" * 60)
