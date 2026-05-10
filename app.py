"""
============================================================================
STUDENT PERFORMANCE ANALYSIS - STREAMLIT DASHBOARD
============================================================================
Interactive web application for exploring student performance data.
Features: Data upload, EDA visualizations, statistical tests,
outlier detection, and linear regression predictions.

Run with: streamlit run app.py
============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from statsmodels.stats.weightstats import ztest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #43A047;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .info-text {
        font-size: 0.95rem;
        color: #424242;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("🎓 Navigation")
page = st.sidebar.radio(
    "Select a Section:",
    ["📊 Dataset Overview", "📈 EDA Visualizations", "📐 Statistical Analysis",
     "🔍 Outlier Detection", "🤖 Regression Model", "📋 Summary"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**About this Dashboard**

This interactive dashboard provides a comprehensive analysis 
of student performance data including EDA, statistical tests, 
outlier detection, and linear regression modeling.

**Dataset**: 500 students, 11 features
""")

# ============================================================
# DATA LOADING FUNCTION
# ============================================================
EDU_MAP = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}

def generate_sample_data(n=500, seed=42):
    """Create a reproducible dataset when no spreadsheet is available."""
    rng = np.random.default_rng(seed)
    study_hours = rng.normal(15, 5, n).clip(2, 30).round(1)
    attendance = rng.normal(76, 12, n).clip(50, 100).round(1)
    previous_score = rng.normal(65, 12, n).clip(30, 95).round(1)
    parent_education = rng.choice(
        list(EDU_MAP.keys()),
        n,
        p=[0.35, 0.35, 0.22, 0.08]
    )
    final_score = (
        8
        + study_hours * 1.65
        + attendance * 0.22
        + previous_score * 0.22
        + pd.Series(parent_education).map(EDU_MAP).to_numpy() * 1.5
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

@st.cache_data
def load_data():
    """Load and cache the student performance dataset."""
    data_path = Path('student_performance.xlsx')
    if data_path.exists():
        return pd.read_excel(data_path)
    return generate_sample_data()

# Try to load data
df = load_data()
uploaded_file = st.sidebar.file_uploader("Use a different dataset", type=['xlsx', 'csv'])
if uploaded_file is not None:
    if uploaded_file.name.lower().endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def encode_data(df):
    """Encode categorical variables for analysis."""
    df_enc = df.copy()
    df_enc['gender'] = df_enc['gender'].map({'Male': 0, 'Female': 1})
    df_enc['internet_access'] = df_enc['internet_access'].map({'No': 0, 'Yes': 1})
    df_enc['extracurricular'] = df_enc['extracurricular'].map({'No': 0, 'Yes': 1})
    df_enc['passed'] = df_enc['passed'].map({'No': 0, 'Yes': 1})
    edu_map = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
    df_enc['parent_education'] = df_enc['parent_education'].map(edu_map)
    return df_enc

def get_model():
    """Train and return the linear regression model."""
    df_clean = df.copy()
    df_clean['parent_education'] = df_clean['parent_education'].fillna(df_clean['parent_education'].mode()[0])
    df_enc = pd.get_dummies(df_clean, columns=['gender', 'internet_access', 'extracurricular'], drop_first=True)
    df_enc['parent_education'] = df_enc['parent_education'].map(EDU_MAP)
    df_enc['passed'] = df_enc['passed'].map({'No': 0, 'Yes': 1})

    X = df_enc.drop(columns=['student_id', 'passed', 'final_score'])
    y = df_enc['final_score']
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)

    return model, scaler, X.columns.tolist(), X_test_s, y_test

# ============================================================
# PAGE 1: DATASET OVERVIEW
# ============================================================
if page == "📊 Dataset Overview":
    st.markdown('<div class="main-header">📊 Dataset Overview</div>', unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Students", len(df))
    col2.metric("Features", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    col4.metric("Duplicates", df.duplicated().sum())
    col5.metric("Pass Rate", f"{(df['passed'] == 'Yes').mean() * 100:.1f}%")

    st.markdown("---")

    # Dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # Data types
    st.subheader("Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.values,
        'Non-Null Count': df.count().values,
        'Missing %': (df.isnull().sum() / len(df) * 100).values
    })
    st.dataframe(col_info, use_container_width=True)

    # Descriptive statistics
    st.subheader("Descriptive Statistics")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    st.dataframe(df[numerical_cols].describe().round(2), use_container_width=True)

    # Categorical summary
    st.subheader("Categorical Variables Summary")
    cat_cols = ['gender', 'parent_education', 'internet_access', 'extracurricular', 'passed']
    for col in cat_cols:
        st.write(f"**{col}**")
        st.write(df[col].value_counts())

# ============================================================
# PAGE 2: EDA VISUALIZATIONS
# ============================================================
elif page == "📈 EDA Visualizations":
    st.markdown('<div class="main-header">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    viz_type = st.sidebar.selectbox(
        "Select Visualization:",
        ["Histograms", "Boxplots", "Countplots", "Correlation Heatmap",
         "Pairplot", "Scatterplots"]
    )

    numerical_cols = ['age', 'study_hours_per_week', 'attendance_rate', 'previous_score', 'final_score']

    if viz_type == "Histograms":
        st.subheader("Distribution Histograms")
        selected_col = st.selectbox("Select variable:", numerical_cols)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df[selected_col], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(df[selected_col].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[selected_col].mean():.1f}')
        ax.axvline(df[selected_col].median(), color='green', linestyle='-.', linewidth=2, label=f'Median: {df[selected_col].median():.1f}')
        ax.set_xlabel(selected_col)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {selected_col}')
        ax.legend()
        st.pyplot(fig)

    elif viz_type == "Boxplots":
        st.subheader("Boxplots")
        selected_col = st.selectbox("Select variable:", numerical_cols)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(df[selected_col], patch_artist=True,
                   boxprops=dict(facecolor='lightblue', color='navy'),
                   medianprops=dict(color='red', linewidth=2))
        ax.set_title(f'Boxplot of {selected_col}')
        ax.set_ylabel(selected_col)
        st.pyplot(fig)

    elif viz_type == "Countplots":
        st.subheader("Countplots")
        cat_cols = ['gender', 'parent_education', 'internet_access', 'extracurricular', 'passed']
        selected_col = st.selectbox("Select variable:", cat_cols)
        fig, ax = plt.subplots(figsize=(8, 5))
        value_counts = df[selected_col].value_counts()
        colors = sns.color_palette("husl", len(value_counts))
        bars = ax.bar(value_counts.index.astype(str), value_counts.values, color=colors, edgecolor='black')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                   ha='center', va='bottom')
        ax.set_title(f'Distribution of {selected_col}')
        ax.set_xlabel(selected_col)
        ax.set_ylabel('Count')
        st.pyplot(fig)

    elif viz_type == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        df_enc = encode_data(df)
        df_corr = df_enc.drop('student_id', axis=1)
        corr_matrix = df_corr.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                    square=True, linewidths=0.5, mask=mask, ax=ax)
        ax.set_title('Correlation Matrix Heatmap')
        st.pyplot(fig)

        st.subheader("Correlation with Final Score")
        st.write(corr_matrix['final_score'].sort_values(ascending=False))

    elif viz_type == "Pairplot":
        st.subheader("Pairplot")
        pair_cols = st.multiselect(
            "Select variables for pairplot:",
            numerical_cols,
            default=['study_hours_per_week', 'attendance_rate', 'final_score']
        )
        if len(pair_cols) >= 2:
            fig = sns.pairplot(df[pair_cols], diag_kind='kde', plot_kws={'alpha': 0.6})
            st.pyplot(fig)
        else:
            st.warning("Please select at least 2 variables.")

    elif viz_type == "Scatterplots":
        st.subheader("Scatterplots")
        x_var = st.selectbox("X variable:", numerical_cols, index=1)
        y_var = st.selectbox("Y variable:", numerical_cols, index=4)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(data=df, x=x_var, y=y_var, ax=ax,
                    scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
        ax.set_title(f'{x_var} vs {y_var}')
        st.pyplot(fig)

        # Calculate correlation
        r, p = stats.pearsonr(df[x_var], df[y_var])
        st.info(f"Pearson correlation: r = {r:.4f}, p-value = {p:.6f}")

# ============================================================
# PAGE 3: STATISTICAL ANALYSIS
# ============================================================
elif page == "📐 Statistical Analysis":
    st.markdown('<div class="main-header">📐 Statistical Analysis</div>', unsafe_allow_html=True)

    test_type = st.sidebar.selectbox(
        "Select Statistical Test:",
        ["Z-Test", "T-Test (Independent)", "T-Test (One Sample)",
         "Pearson Correlation", "Chi-Square Test", "ANOVA"]
    )

    if test_type == "Z-Test":
        st.subheader("Z-Test: Final Score vs Hypothesized Mean")
        hypothesized_mean = st.number_input("Hypothesized mean:", value=50.0)
        z_stat, z_pvalue = ztest(df['final_score'], value=hypothesized_mean)

        col1, col2, col3 = st.columns(3)
        col1.metric("Sample Mean", f"{df['final_score'].mean():.2f}")
        col2.metric("Z-Statistic", f"{z_stat:.4f}")
        col3.metric("P-Value", f"{z_pvalue:.6f}")

        st.write(f"**H0**: Mean final score = {hypothesized_mean}")
        st.write(f"**H1**: Mean final score != {hypothesized_mean}")
        st.write(f"**Result**: {'Reject H0 (significant)' if z_pvalue < 0.05 else 'Fail to reject H0 (not significant)'}")

    elif test_type == "T-Test (Independent)":
        st.subheader("Independent T-Test")
        group_col = st.selectbox("Grouping variable:", ['gender', 'internet_access', 'extracurricular'])
        group1 = df[df[group_col] == df[group_col].unique()[0]]['final_score']
        group2 = df[df[group_col] == df[group_col].unique()[1]]['final_score']

        t_stat, t_pvalue = stats.ttest_ind(group1, group2)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"Group 1 Mean", f"{group1.mean():.2f}")
        col2.metric(f"Group 2 Mean", f"{group2.mean():.2f}")
        col3.metric("T-Statistic", f"{t_stat:.4f}")
        col4.metric("P-Value", f"{t_pvalue:.6f}")

        st.write(f"**H0**: Mean scores are equal between groups")
        st.write(f"**H1**: Mean scores differ between groups")
        st.write(f"**Result**: {'Reject H0 (significant)' if t_pvalue < 0.05 else 'Fail to reject H0 (not significant)'}")

    elif test_type == "T-Test (One Sample)":
        st.subheader("One-Sample T-Test")
        test_value = st.number_input("Test value:", value=60.0)
        t_stat, t_pvalue = stats.ttest_1samp(df['final_score'], test_value)

        col1, col2, col3 = st.columns(3)
        col1.metric("Sample Mean", f"{df['final_score'].mean():.2f}")
        col2.metric("T-Statistic", f"{t_stat:.4f}")
        col3.metric("P-Value", f"{t_pvalue:.6f}")

        st.write(f"**H0**: Mean = {test_value}")
        st.write(f"**H1**: Mean != {test_value}")
        st.write(f"**Result**: {'Reject H0 (significant)' if t_pvalue < 0.05 else 'Fail to reject H0 (not significant)'}")

    elif test_type == "Pearson Correlation":
        st.subheader("Pearson Correlation Test")
        var1 = st.selectbox("Variable 1:", numerical_cols, index=1)
        var2 = st.selectbox("Variable 2:", numerical_cols, index=4)
        r, p = stats.pearsonr(df[var1], df[var2])

        col1, col2 = st.columns(2)
        col1.metric("Correlation (r)", f"{r:.4f}")
        col2.metric("P-Value", f"{p:.6f}")

        st.write(f"**H0**: No linear correlation (r = 0)")
        st.write(f"**H1**: Linear correlation exists (r != 0)")
        st.write(f"**Strength**: {'Strong' if abs(r) > 0.7 else 'Moderate' if abs(r) > 0.3 else 'Weak'} {'positive' if r > 0 else 'negative'}")

    elif test_type == "Chi-Square Test":
        st.subheader("Chi-Square Test of Independence")
        var1 = st.selectbox("Variable 1:", ['gender', 'internet_access', 'extracurricular'])
        var2 = st.selectbox("Variable 2:", ['passed'], index=0)
        contingency = pd.crosstab(df[var1], df[var2])
        chi2, p, dof, expected = stats.chi2_contingency(contingency)

        st.write("Contingency Table:")
        st.dataframe(contingency)

        col1, col2, col3 = st.columns(3)
        col1.metric("Chi-Square", f"{chi2:.4f}")
        col2.metric("P-Value", f"{p:.6f}")
        col3.metric("DoF", dof)

        st.write(f"**H0**: Variables are independent")
        st.write(f"**H1**: Variables are dependent")
        st.write(f"**Result**: {'Reject H0 (dependent)' if p < 0.05 else 'Fail to reject H0 (independent)'}")

    elif test_type == "ANOVA":
        st.subheader("One-Way ANOVA")
        df_anova = df.dropna(subset=['parent_education'])
        groups = [g['final_score'].values for _, g in df_anova.groupby('parent_education')]
        f_stat, p_value = stats.f_oneway(*groups)

        st.write("Group Means:")
        for name, group in df_anova.groupby('parent_education'):
            st.write(f"- {name}: {group['final_score'].mean():.2f}")

        col1, col2 = st.columns(2)
        col1.metric("F-Statistic", f"{f_stat:.4f}")
        col2.metric("P-Value", f"{p_value:.6f}")

        st.write(f"**H0**: All group means are equal")
        st.write(f"**H1**: At least one group mean differs")
        st.write(f"**Result**: {'Reject H0 (significant)' if p_value < 0.05 else 'Fail to reject H0 (not significant)'}")

# ============================================================
# PAGE 4: OUTLIER DETECTION
# ============================================================
elif page == "🔍 Outlier Detection":
    st.markdown('<div class="main-header">🔍 Outlier Detection</div>', unsafe_allow_html=True)

    method = st.sidebar.selectbox(
        "Select Method:",
        ["IQR Method", "Z-Score Method", "Visualization"]
    )

    numerical_cols = ['age', 'study_hours_per_week', 'attendance_rate', 'previous_score', 'final_score']
    selected_col = st.selectbox("Select variable:", numerical_cols, index=4)

    if method == "IQR Method":
        st.subheader("IQR Method (Interquartile Range)")
        Q1 = df[selected_col].quantile(0.25)
        Q3 = df[selected_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Q1 (25%)", f"{Q1:.2f}")
        col2.metric("Q3 (75%)", f"{Q3:.2f}")
        col3.metric("IQR", f"{IQR:.2f}")
        col4.metric("Outliers Found", len(outliers))

        st.write(f"**Lower Bound**: {lower_bound:.2f}")
        st.write(f"**Upper Bound**: {upper_bound:.2f}")

        if len(outliers) > 0:
            st.write("Outlier records:")
            st.dataframe(outliers[['student_id', selected_col]])
        else:
            st.success("No outliers detected using IQR method!")

    elif method == "Z-Score Method":
        st.subheader("Z-Score Method (Threshold = |3|)")
        z_scores = np.abs(stats.zscore(df[selected_col]))
        threshold = st.slider("Z-score threshold:", 1.0, 5.0, 3.0, 0.5)
        outliers = df[z_scores > threshold]

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean", f"{df[selected_col].mean():.2f}")
        col2.metric("Std Dev", f"{df[selected_col].std():.2f}")
        col3.metric("Outliers Found", len(outliers))

        if len(outliers) > 0:
            st.write("Outlier records:")
            st.dataframe(outliers[['student_id', selected_col]])
        else:
            st.success(f"No outliers detected with Z-score threshold of {threshold}!")

    elif method == "Visualization":
        st.subheader("Outlier Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(df[selected_col], patch_artist=True,
                   boxprops=dict(facecolor='lightblue', color='navy'),
                   medianprops=dict(color='red', linewidth=2))
        ax.set_title(f'Boxplot of {selected_col}')
        ax.set_ylabel(selected_col)
        st.pyplot(fig)

# ============================================================
# PAGE 5: REGRESSION MODEL
# ============================================================
elif page == "🤖 Regression Model":
    st.markdown('<div class="main-header">🤖 Linear Regression Model</div>', unsafe_allow_html=True)

    model_tab, predict_tab = st.tabs(["Model Evaluation", "Make Prediction"])

    with model_tab:
        st.subheader("Model Training & Evaluation")

        with st.spinner("Training model..."):
            model, scaler, feature_names, X_test_s, y_test = get_model()
            y_pred = model.predict(X_test_s)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("MSE", f"{mse:.2f}")
        col3.metric("RMSE", f"{rmse:.2f}")
        col4.metric("R² Score", f"{r2:.4f}")

        # Actual vs Predicted
        st.subheader("Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, alpha=0.6, color='green', edgecolors='black')
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Final Score')
        ax.set_ylabel('Predicted Final Score')
        ax.set_title(f'Actual vs Predicted (R² = {r2:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Feature Importance
        st.subheader("Feature Importance")
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_,
            'Abs_Coefficient': np.abs(model.coef_)
        }).sort_values('Abs_Coefficient', ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(coef_df)))
        ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
        ax.set_xlabel('Coefficient Value')
        ax.set_title('Feature Coefficients')
        ax.grid(True, alpha=0.3, axis='x')
        st.pyplot(fig)

    with predict_tab:
        st.subheader("Predict Final Score")
        st.write("Enter student details to predict their final score:")

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age:", min_value=15, max_value=19, value=17)
            study_hours = st.number_input("Study Hours per Week:", min_value=2, max_value=30, value=15)
            attendance = st.slider("Attendance Rate (%):", 50.0, 100.0, 76.0)
            prev_score = st.number_input("Previous Score:", min_value=30, max_value=95, value=65)

        with col2:
            parent_edu = st.selectbox("Parent Education:", ['High School', 'Bachelor', 'Master', 'PhD'])
            gender = st.selectbox("Gender:", ['Female', 'Male'])
            internet = st.selectbox("Internet Access:", ['No', 'Yes'])
            extra = st.selectbox("Extracurricular:", ['No', 'Yes'])

        if st.button("Predict Final Score", type="primary"):
            # Prepare input
            input_data = pd.DataFrame([{
                'age': age,
                'study_hours_per_week': study_hours,
                'attendance_rate': attendance,
                'parent_education': EDU_MAP[parent_edu],
                'previous_score': prev_score,
                'gender_Male': 1 if gender == 'Male' else 0,
                'internet_access_Yes': 1 if internet == 'Yes' else 0,
                'extracurricular_Yes': 1 if extra == 'Yes' else 0
            }], columns=feature_names)
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]

            st.success(f"Predicted Final Score: {prediction:.1f}")
            if prediction >= 60:
                st.balloons()
                st.info("This student is predicted to PASS! ✅")
            else:
                st.warning("This student is predicted to FAIL. ⚠️ Additional support recommended.")

# ============================================================
# PAGE 6: SUMMARY
# ============================================================
elif page == "📋 Summary":
    st.markdown('<div class="main-header">📋 Project Summary</div>', unsafe_allow_html=True)

    st.markdown("""
    ## Key Findings

    ### Most Influential Factors for Student Performance:
    1. **Study Hours per Week** (r = 0.804) - *Strongest predictor*
    2. **Attendance Rate** (r = 0.241) - *Moderate predictor*
    3. **Previous Score** (r = 0.165) - *Weak predictor*

    ### Statistical Insights:
    - Mean final score (55.98) is significantly below typical passing threshold (60)
    - No significant gender difference in performance (p = 0.485)
    - No significant effect of internet access (p = 0.259)
    - Study hours show very strong positive correlation with final scores

    ### Model Performance:
    - **R² Score**: 0.753 (75.3% variance explained)
    - **RMSE**: 7.32 points
    - **MAE**: 6.26 points
    - The model performs well with no significant overfitting

    ### Recommendations:
    - **For Students**: Increase weekly study hours (strongest impact)
    - **For Educators**: Monitor attendance, provide support for struggling students
    - **For Policy**: Gender-neutral approaches are statistically justified

    ### Limitations:
    - Synthetic dataset may not reflect real-world complexity
    - 23.4% missing parent education data
    - Linear model assumes linear relationships only
    """)

    st.markdown("---")
    st.info("This analysis was generated as part of a university-level data science project. All statistical tests use a significance level of α = 0.05.")
