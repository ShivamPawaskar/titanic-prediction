import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="T",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_training_data() -> pd.DataFrame:
    return pd.read_csv("train.csv")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["Title"] = data["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False).fillna("Unknown")
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

    rare_titles = {
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    }
    data["Title"] = data["Title"].replace(
        {
            "Mlle": "Miss",
            "Ms": "Miss",
            "Mme": "Mrs",
        }
    )
    data["Title"] = data["Title"].apply(lambda value: "Rare" if value in rare_titles else value)
    return data


@st.cache_resource
def train_pipeline():
    train = build_features(load_training_data())
    feature_cols = [
        "Pclass",
        "Sex",
        "Age",
        "Fare",
        "Embarked",
        "FamilySize",
        "IsAlone",
        "Title",
    ]
    X = train[feature_cols]
    y = train["Survived"]

    numeric_features = ["Age", "Fare", "FamilySize", "IsAlone"]
    categorical_features = ["Pclass", "Sex", "Embarked", "Title"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(X, y)
    return model, train


def probability_chart(probability: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Survival Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#d1495b"},
                "steps": [
                    {"range": [0, 35], "color": "#f4d6cc"},
                    {"range": [35, 65], "color": "#f5d547"},
                    {"range": [65, 100], "color": "#6fb98f"},
                ],
            },
        )
    )
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def build_input_frame(
    pclass: int,
    sex: str,
    age: float,
    sibsp: int,
    parch: int,
    fare: float,
    embarked: str,
    title: str,
) -> pd.DataFrame:
    sample = pd.DataFrame(
        [
            {
                "Name": f"Passenger, {title}. Demo",
                "Pclass": pclass,
                "Sex": sex,
                "Age": age,
                "SibSp": sibsp,
                "Parch": parch,
                "Fare": fare,
                "Embarked": embarked,
            }
        ]
    )
    sample = build_features(sample)
    return sample


st.markdown(
    """
    <style>
    .hero-card {
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(209,73,91,0.12), rgba(245,213,71,0.18));
        border: 1px solid rgba(209,73,91,0.16);
        margin-bottom: 1rem;
    }
    .small-note {
        color: #5f5f5f;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

model, train_df = train_pipeline()
avg_survival = train_df["Survived"].mean()
female_survival = train_df.loc[train_df["Sex"] == "female", "Survived"].mean()
male_survival = train_df.loc[train_df["Sex"] == "male", "Survived"].mean()

st.markdown(
    """
    <div class="hero-card">
        <h1 style="margin-bottom:0.3rem;">Titanic Survival Predictor</h1>
        <p style="margin-bottom:0.2rem;">
            Interactive demo built on the Kaggle Titanic dataset with lightweight feature engineering
            and a live classification pipeline.
        </p>
        <p class="small-note">
            Use the sidebar to create a passenger profile and compare the prediction against the
            historical survival patterns in the training data.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Training Rows", f"{len(train_df)}")
metric_col2.metric("Average Survival Rate", f"{avg_survival:.1%}")
metric_col3.metric("Women vs Men", f"{female_survival:.1%} / {male_survival:.1%}")

st.sidebar.header("Passenger Profile")
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3], index=2)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", min_value=0.0, max_value=80.0, value=29.0, step=1.0)
sibsp = st.sidebar.slider("Siblings / Spouses Aboard", min_value=0, max_value=8, value=0, step=1)
parch = st.sidebar.slider("Parents / Children Aboard", min_value=0, max_value=6, value=0, step=1)
fare = st.sidebar.slider("Fare", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
embarked = st.sidebar.selectbox("Embarked", ["S", "C", "Q"])
title = st.sidebar.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Dr", "Rare"])

sample = build_input_frame(pclass, sex, age, sibsp, parch, fare, embarked, title)
feature_cols = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Embarked",
    "FamilySize",
    "IsAlone",
    "Title",
]
survival_prob = model.predict_proba(sample[feature_cols])[0, 1]
predicted_label = "Survived" if survival_prob >= 0.5 else "Did Not Survive"

left_col, right_col = st.columns([1.1, 0.9])

with left_col:
    st.subheader("Prediction")
    st.plotly_chart(probability_chart(survival_prob), use_container_width=True)
    st.success(f"Predicted Outcome: {predicted_label}")
    st.caption(
        "This app uses a clean logistic regression pipeline for instant predictions. "
        "The full notebook explores stronger ensemble models for competition-style performance."
    )

with right_col:
    st.subheader("Passenger Summary")
    st.dataframe(
        sample[
            [
                "Pclass",
                "Sex",
                "Age",
                "Fare",
                "Embarked",
                "Title",
                "FamilySize",
                "IsAlone",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    survival_gap = survival_prob - avg_survival
    st.metric("Difference vs Dataset Baseline", f"{survival_gap:+.1%}")

insight_col1, insight_col2 = st.columns(2)

with insight_col1:
    st.subheader("Historical Survival by Sex")
    sex_rates = (
        train_df.groupby("Sex")["Survived"]
        .mean()
        .reindex(["female", "male"])
        .rename(index={"female": "Female", "male": "Male"})
    )
    fig = go.Figure(
        data=[
            go.Bar(
                x=sex_rates.index.tolist(),
                y=sex_rates.values.tolist(),
                marker_color=["#6fb98f", "#d1495b"],
            )
        ]
    )
    fig.update_layout(
        yaxis_title="Survival Rate",
        yaxis_tickformat=".0%",
        height=320,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

with insight_col2:
    st.subheader("Historical Survival by Passenger Class")
    class_rates = train_df.groupby("Pclass")["Survived"].mean().sort_index()
    fig = go.Figure(
        data=[
            go.Bar(
                x=[f"Class {x}" for x in class_rates.index.tolist()],
                y=class_rates.values.tolist(),
                marker_color=["#edae49", "#00798c", "#d1495b"],
            )
        ]
    )
    fig.update_layout(
        yaxis_title="Survival Rate",
        yaxis_tickformat=".0%",
        height=320,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("How To Run")
st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
