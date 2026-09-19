import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EduPredict",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.info-box {
    background-color: #eef4ff;
    padding: 18px;
    border-radius: 10px;
    border-left: 5px solid #4f46e5;
}

.prediction-container {
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #dfe3e8;
    text-align: center;
    background-color: #ffffff;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.title("🎓 EduPredict")

st.subheader(
    "Student Academic Performance Prediction Dashboard"
)

st.markdown(
    """
    <div class="info-box">
    Upload your Student Performance CSV file from the sidebar
    to begin the analysis and prediction.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎓 EduPredict")

st.sidebar.markdown(
    "### Student Academic Performance"
)

st.sidebar.write(
    "Analyze student performance and predict examination "
    "scores using machine learning."
)

st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Student Dataset",
    type=["csv"],
    help="Upload your Student Academic Performance CSV file."
)


# =========================================================
# WAIT UNTIL CSV IS UPLOADED
# =========================================================

if uploaded_file is None:

    st.info(
        "👈 Please upload your Student Performance CSV file "
        "from the sidebar to start the dashboard."
    )

    st.markdown("### How to use EduPredict")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("""
        ### 1️⃣ Upload Dataset

        Click **Browse files** in the sidebar and select
        your Student Performance CSV file.
        """)

    with col2:

        st.markdown("""
        ### 2️⃣ Explore Analytics

        View score distributions, attendance relationships,
        categorical analysis and correlations.
        """)

    with col3:

        st.markdown("""
        ### 3️⃣ Predict Score

        Enter student information and use the machine
        learning model to estimate the exam score.
        """)

    st.stop()


# =========================================================
# LOAD CSV
# =========================================================

try:

    uploaded_file.seek(0)

    df = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(
        f"❌ Unable to read the uploaded CSV file: {e}"
    )

    st.stop()


# =========================================================
# DATASET VALIDATION
# =========================================================

if "Exam_Score" not in df.columns:

    st.error(
        "❌ Invalid dataset. The CSV must contain "
        "'Exam_Score' column."
    )

    st.write("### Columns found in the uploaded file:")

    st.write(df.columns.tolist())

    st.stop()


# =========================================================
# CLEAN TARGET
# =========================================================

df["Exam_Score"] = pd.to_numeric(
    df["Exam_Score"],
    errors="coerce"
)

df = df.dropna(
    subset=["Exam_Score"]
).copy()


# =========================================================
# SUCCESS MESSAGE
# =========================================================

st.success(
    f"✅ Dataset uploaded successfully — "
    f"{len(df):,} student records loaded."
)


# =========================================================
# DATASET STATISTICS
# =========================================================

total_students = len(df)

total_columns = len(df.columns)

total_features = total_columns - 1

missing_values = int(
    df.isnull().sum().sum()
)

average_score = df["Exam_Score"].mean()

duplicate_rows = int(
    df.duplicated().sum()
)


# =========================================================
# OVERVIEW METRICS
# =========================================================

st.markdown("## 📊 Dataset Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "Students",
        f"{total_students:,}"
    )

with col2:

    st.metric(
        "Features",
        total_features
    )

with col3:

    st.metric(
        "Average Score",
        f"{average_score:.2f}"
    )

with col4:

    st.metric(
        "Missing Values",
        f"{missing_values:,}"
    )

with col5:

    st.metric(
        "Duplicates",
        duplicate_rows
    )


st.write("")


# =========================================================
# FEATURES AND TARGET
# =========================================================

X = df.drop(
    columns=["Exam_Score"]
)

y = df["Exam_Score"]


# Remove ID if present
if "ID" in X.columns:

    X = X.drop(
        columns=["ID"]
    )


# =========================================================
# IDENTIFY NUMERIC AND CATEGORICAL FEATURES
# =========================================================

numeric_features = X.select_dtypes(
    include=[
        "int64",
        "float64",
        "int32",
        "float32"
    ]
).columns.tolist()


categorical_features = X.select_dtypes(
    include=[
        "object",
        "category",
        "bool"
    ]
).columns.tolist()


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# =========================================================
# PREPROCESSING
# =========================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# =========================================================
# RANDOM FOREST MODEL
# =========================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=150,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# =========================================================
# TRAIN MODEL
# =========================================================

with st.spinner(
    "🤖 Training Random Forest model..."
):

    model.fit(
        X_train,
        y_train
    )


# =========================================================
# MODEL PREDICTIONS
# =========================================================

y_pred = model.predict(
    X_test
)


# =========================================================
# MODEL EVALUATION
# =========================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Overview",
        "📈 Performance Analytics",
        "🎯 Score Predictor",
        "🤖 Model Performance"
    ]
)


# =========================================================
# TAB 1 — OVERVIEW
# =========================================================

with tab1:

    st.header(
        "📊 Student Performance Overview"
    )

    col1, col2 = st.columns(2)


    # -----------------------------------------------------
    # SCORE DISTRIBUTION
    # -----------------------------------------------------

    with col1:

        st.subheader(
            "Exam Score Distribution"
        )

        fig, ax = plt.subplots()

        ax.hist(
            df["Exam_Score"],
            bins=20,
            edgecolor="black"
        )

        ax.set_xlabel(
            "Exam Score"
        )

        ax.set_ylabel(
            "Number of Students"
        )

        ax.set_title(
            "Distribution of Exam Scores"
        )

        st.pyplot(fig)

        plt.close(fig)


    # -----------------------------------------------------
    # ATTENDANCE VS SCORE
    # -----------------------------------------------------

    with col2:

        st.subheader(
            "Attendance vs Exam Score"
        )

        if "Attendance" in df.columns:

            fig, ax = plt.subplots()

            ax.scatter(
                df["Attendance"],
                df["Exam_Score"],
                alpha=0.5
            )

            ax.set_xlabel(
                "Attendance"
            )

            ax.set_ylabel(
                "Exam Score"
            )

            ax.set_title(
                "Attendance vs Exam Score"
            )

            st.pyplot(fig)

            plt.close(fig)

        else:

            st.info(
                "Attendance column not found."
            )


    st.divider()


    # -----------------------------------------------------
    # DATASET PREVIEW
    # -----------------------------------------------------

    st.subheader(
        "📋 Dataset Preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# =========================================================
# TAB 2 — PERFORMANCE ANALYTICS
# =========================================================

with tab2:

    st.header(
        "📈 Performance Analytics"
    )


    # -----------------------------------------------------
    # CATEGORICAL ANALYSIS
    # -----------------------------------------------------

    st.subheader(
        "Performance by Category"
    )

    if categorical_features:

        selected_category = st.selectbox(
            "Select a categorical factor:",
            categorical_features
        )

        category_summary = (
            df.groupby(
                selected_category
            )["Exam_Score"]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        st.bar_chart(
            category_summary
        )

        st.dataframe(
            category_summary.to_frame(
                "Average Exam Score"
            ),
            use_container_width=True
        )

    else:

        st.info(
            "No categorical columns available."
        )


    st.divider()


    # -----------------------------------------------------
    # NUMERICAL ANALYSIS
    # -----------------------------------------------------

    st.subheader(
        "Numerical Factor vs Exam Score"
    )

    if numeric_features:

        selected_numeric = st.selectbox(
            "Select numerical factor:",
            numeric_features
        )

        fig, ax = plt.subplots()

        ax.scatter(
            df[selected_numeric],
            df["Exam_Score"],
            alpha=0.5
        )

        ax.set_xlabel(
            selected_numeric
        )

        ax.set_ylabel(
            "Exam Score"
        )

        ax.set_title(
            f"{selected_numeric} vs Exam Score"
        )

        st.pyplot(fig)

        plt.close(fig)


    st.divider()


    # -----------------------------------------------------
    # CORRELATION
    # -----------------------------------------------------

    st.subheader(
        "🔗 Correlation with Exam Score"
    )

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if "Exam_Score" in numeric_df.columns:

        correlations = (
            numeric_df.corr()["Exam_Score"]
            .drop("Exam_Score")
            .sort_values(
                ascending=False
            )
        )

        st.dataframe(
            correlations.to_frame(
                name="Correlation"
            ),
            use_container_width=True
        )


# =========================================================
# TAB 3 — SCORE PREDICTOR
# =========================================================

with tab3:

    st.header(
        "🎯 Individual Student Score Predictor"
    )

    st.write(
        "Enter the student's information below "
        "to estimate the expected examination score."
    )


    # -----------------------------------------------------
    # INPUT DATA
    # -----------------------------------------------------

    input_data = {}


    # -----------------------------------------------------
    # NUMERICAL INPUTS
    # -----------------------------------------------------

    if numeric_features:

        st.subheader(
            "📚 Academic & Numerical Information"
        )

        numeric_cols = st.columns(2)


        for i, column in enumerate(
            numeric_features
        ):

            with numeric_cols[i % 2]:

                values = pd.to_numeric(
                    X[column],
                    errors="coerce"
                )

                min_value = values.min()

                max_value = values.max()

                median_value = values.median()


                if pd.isna(min_value):
                    min_value = 0

                if pd.isna(max_value):
                    max_value = 100

                if pd.isna(median_value):
                    median_value = min_value


                if pd.api.types.is_integer_dtype(
                    X[column]
                ):

                    input_data[column] = st.number_input(
                        column,
                        min_value=int(min_value),
                        max_value=int(max_value),
                        value=int(median_value),
                        step=1
                    )

                else:

                    input_data[column] = st.number_input(
                        column,
                        min_value=float(min_value),
                        max_value=float(max_value),
                        value=float(median_value),
                        step=0.1
                    )


    # -----------------------------------------------------
    # CATEGORICAL INPUTS
    # -----------------------------------------------------

    if categorical_features:

        st.subheader(
            "👨‍👩‍👧 Student Background Information"
        )

        categorical_cols = st.columns(2)


        for i, column in enumerate(
            categorical_features
        ):

            with categorical_cols[i % 2]:

                values = (
                    df[column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                values = sorted(
                    values
                )


                if values:

                    input_data[column] = st.selectbox(
                        column,
                        values
                    )


    st.write("")


    # =====================================================
    # PREDICT BUTTON
    # =====================================================

    if st.button(
        "🚀 Predict Exam Score",
        use_container_width=True
    ):

        input_df = pd.DataFrame(
            [input_data]
        )


        try:

            prediction = model.predict(
                input_df
            )[0]


            # Keep prediction between 0 and 100
            prediction = max(
                0,
                min(
                    100,
                    prediction
                )
            )


            # =================================================
            # PREDICTION RESULT
            # =================================================

            st.markdown(
                "## 🎯 Prediction Result"
            )


            col1, col2, col3 = st.columns(
                [1, 2, 1]
            )


            with col2:

                st.metric(
                    label="Predicted Exam Score",
                    value=f"{prediction:.2f} / 100"
                )

                st.caption(
                    "Expected score based on the trained "
                    "Random Forest model."
                )


            st.write("")


            # =================================================
            # PERFORMANCE LEVEL
            # =================================================

            if prediction >= 90:

                level = "Excellent 🏆"

                message = (
                    "Outstanding expected performance. "
                    "Continue maintaining the current study habits."
                )

            elif prediction >= 80:

                level = "Very Good ⭐"

                message = (
                    "Strong expected performance. "
                    "Consistent preparation should help maintain this level."
                )

            elif prediction >= 70:

                level = "Good 👍"

                message = (
                    "Good expected performance. "
                    "Some additional academic effort may improve the score."
                )

            elif prediction >= 60:

                level = "Average 📚"

                message = (
                    "Moderate expected performance. "
                    "Regular study and improvement in weak areas are recommended."
                )

            else:

                level = "Needs Improvement 📖"

                message = (
                    "The predicted score is relatively low. "
                    "Additional academic support and consistent study are recommended."
                )


            st.success(
                f"**Performance Level:** {level}"
            )


            st.info(
                f"💡 **Recommendation:** {message}"
            )


        except Exception as e:

            st.error(
                f"❌ Prediction failed: {e}"
            )


# =========================================================
# TAB 4 — MODEL PERFORMANCE
# =========================================================

with tab4:

    st.header(
        "🤖 Machine Learning Model Performance"
    )


    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "MAE",
            f"{mae:.3f}"
        )


    with col2:

        st.metric(
            "RMSE",
            f"{rmse:.3f}"
        )


    with col3:

        st.metric(
            "R² Score",
            f"{r2:.3f}"
        )


    st.write("")


    st.info(
        """
        **Metric Interpretation**

        **MAE:** Average absolute difference between
        actual and predicted scores.

        **RMSE:** Measures prediction error while giving
        more importance to larger errors.

        **R² Score:** Shows how much variation in exam
        scores is explained by the model.
        """
    )


    st.divider()


    # -----------------------------------------------------
    # ACTUAL VS PREDICTED
    # -----------------------------------------------------

    st.subheader(
        "Actual vs Predicted Exam Scores"
    )


    fig, ax = plt.subplots()


    ax.scatter(
        y_test,
        y_pred,
        alpha=0.5
    )


    min_score = min(
        y_test.min(),
        y_pred.min()
    )

    max_score = max(
        y_test.max(),
        y_pred.max()
    )


    ax.plot(
        [min_score, max_score],
        [min_score, max_score],
        linestyle="--"
    )


    ax.set_xlabel(
        "Actual Exam Score"
    )

    ax.set_ylabel(
        "Predicted Exam Score"
    )

    ax.set_title(
        "Actual vs Predicted Scores"
    )


    st.pyplot(fig)

    plt.close(fig)


    st.divider()


    # -----------------------------------------------------
    # ERROR ANALYSIS
    # -----------------------------------------------------

    st.subheader(
        "Prediction Error Analysis"
    )


    results = pd.DataFrame(
        {
            "Actual Score": y_test.values,
            "Predicted Score": y_pred,
            "Error": y_test.values - y_pred
        }
    )


    results["Absolute Error"] = (
        results["Error"].abs()
    )


    st.dataframe(
        results.head(20),
        use_container_width=True
    )


    st.write(
        f"Average prediction error: "
        f"**{mae:.2f} marks**"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "EduPredict | Student Academic Performance Prediction "
    "System | Machine Learning Capstone Project"
)