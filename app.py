import folium
from folium.plugins import HeatMap
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from streamlit_folium import st_folium

# =========================================================
# PAGE CONFIGURATION & CUSTOM THEME
# =========================================================

st.set_page_config(
    page_title="Crime Analytics & Intelligence System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, modern UI
st.markdown("""
<style>
    :root {
        --surface: rgba(15, 23, 42, 0.74);
        --surface-soft: rgba(30, 41, 59, 0.55);
        --border: rgba(148, 163, 184, 0.16);
        --cyan: #38bdf8;
        --violet: #818cf8;
        --text: #f8fafc;
        --muted: #94a3b8;
    }

    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(56, 189, 248, 0.10), transparent 28%),
            radial-gradient(circle at 88% 4%, rgba(129, 140, 248, 0.10), transparent 30%),
            linear-gradient(180deg, #07101f 0%, #0b1220 48%, #090f1b 100%);
    }

    .block-container {
        max-width: 1480px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3, h4 { letter-spacing: -0.025em; }
    p, label, .stCaption { color: #cbd5e1; }
    
    /* Hero Header */
    .hero-container {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.94) 0%, rgba(30, 41, 59, 0.90) 58%, rgba(30, 27, 75, 0.82) 100%);
        border: 1px solid rgba(129, 140, 248, 0.22);
        padding: 2.3rem 2.6rem;
        border-radius: 22px;
        margin-bottom: 2rem;
        box-shadow: 0 24px 70px -36px rgba(56, 189, 248, 0.55);
    }
    .hero-container::after {
        content: "";
        position: absolute;
        width: 220px;
        height: 220px;
        right: -70px;
        top: -90px;
        border-radius: 50%;
        background: rgba(56, 189, 248, 0.11);
        filter: blur(2px);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        max-width: 800px;
        line-height: 1.5;
        margin-bottom: 0;
    }

    .eyebrow {
        color: #7dd3fc;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.65rem;
    }

    /* Section Cards */
    .section-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Custom Metric Styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.84), rgba(30, 41, 59, 0.62));
        border: 1px solid var(--border);
        padding: 1.2rem 1.35rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
    }

    /* Result Banner Boxes */
    .result-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    .badge-danger { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
    .badge-warning { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
    .badge-success { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
    .badge-info { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid #3b82f6; }

    /* Prediction Result Cards */
    .prediction-box {
        padding: 1.5rem;
        border-radius: 14px;
        margin: 1.2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .prediction-box.high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(15, 23, 42, 0.6));
        border-color: rgba(239, 68, 68, 0.4);
    }
    .prediction-box.medium {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(15, 23, 42, 0.6));
        border-color: rgba(245, 158, 11, 0.4);
    }
    .prediction-box.low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(15, 23, 42, 0.6));
        border-color: rgba(16, 185, 129, 0.4);
    }

    /* Sidebar Refinement */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(7, 16, 31, 0.98), rgba(15, 23, 42, 0.98));
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        border: 1px solid transparent;
        border-radius: 12px;
        padding: 0.55rem 0.65rem;
        margin-bottom: 0.25rem;
        transition: all 0.2s ease;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(56, 189, 248, 0.08);
        border-color: rgba(56, 189, 248, 0.18);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(15, 23, 42, 0.46);
        border-color: var(--border) !important;
        border-radius: 16px;
    }

    .stButton > button {
        min-height: 3rem;
        border-radius: 12px;
        border: 1px solid rgba(56, 189, 248, 0.30);
        font-weight: 750;
        background: linear-gradient(90deg, #0284c7, #4f46e5);
        box-shadow: 0 12px 28px -16px rgba(56, 189, 248, 0.9);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 16px 34px -16px rgba(56, 189, 248, 1);
        border-color: #7dd3fc;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    .stNumberInput input {
        background: rgba(15, 23, 42, 0.72) !important;
        border-color: rgba(148, 163, 184, 0.20) !important;
        border-radius: 10px !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 14px;
        overflow: hidden;
    }

    .model-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1.2rem;
    }
    .model-chip {
        padding: 0.35rem 0.72rem;
        border-radius: 999px;
        border: 1px solid rgba(125, 211, 252, 0.22);
        background: rgba(14, 165, 233, 0.08);
        color: #bae6fd;
        font-size: 0.76rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# HELPER FUNCTIONS & LOGIC
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

def get_time_of_day(hour):
    if pd.isna(hour):
        return "Unknown"
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

def get_season(month):
    if pd.isna(month):
        return "Unknown"
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

def load_optional_artifact(filename, default=None):
    path = BASE_DIR / filename
    if not path.exists():
        return {} if default is None else default
    return joblib.load(path)


def unwrap_pipeline(artifact, possible_keys):
    """Support both the current pipeline files and older dictionary artifacts."""
    if hasattr(artifact, "named_steps"):
        return artifact
    if isinstance(artifact, dict):
        for key in possible_keys:
            candidate = artifact.get(key)
            if hasattr(candidate, "named_steps"):
                return candidate
    raise TypeError("The saved artifact does not contain a scikit-learn pipeline.")


def get_pipeline_features(model, metadata, fallback):
    configured = metadata.get("risk_features") or metadata.get("selected_features")
    if configured:
        return list(configured)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return list(fallback)


def build_model_input(values, required_features):
    missing = [feature for feature in required_features if feature not in values]
    if missing:
        raise ValueError(f"Missing required model inputs: {', '.join(missing)}")
    return pd.DataFrame([{feature: values[feature] for feature in required_features}])


def get_risk_shap_explanation(risk_model, risk_input, risk_features):
    preprocessor = risk_model.named_steps["preprocessor"]
    rf_model = risk_model.named_steps["model"]

    transformed_input = preprocessor.transform(risk_input)

    if hasattr(transformed_input, "toarray"):
        transformed_input = transformed_input.toarray()

    transformed_feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(transformed_input)

    predicted_class = rf_model.predict(transformed_input)[0]
    class_index = list(rf_model.classes_).index(predicted_class)

    if isinstance(shap_values, list):
        current_shap_values = shap_values[class_index][0]
    else:
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            current_shap_values = shap_values[0, :, class_index]
        elif shap_values.ndim == 2:
            current_shap_values = shap_values[0]
        else:
            raise ValueError(f"Unexpected SHAP value shape: {shap_values.shape}")

    feature_contributions = {feature: 0.0 for feature in risk_features}

    for feature_name, shap_value in zip(transformed_feature_names, current_shap_values):
        matched_feature = None
        for original_feature in risk_features:
            if (
                feature_name == f"num__{original_feature}"
                or feature_name == f"cat__{original_feature}"
                or feature_name.startswith(f"cat__{original_feature}_")
            ):
                matched_feature = original_feature
                break
        if matched_feature is not None:
            feature_contributions[matched_feature] += float(shap_value)

    shap_df = pd.DataFrame(
        {
            "Feature": list(feature_contributions.keys()),
            "Contribution": list(feature_contributions.values())
        }
    )

    shap_df["Absolute Contribution"] = shap_df["Contribution"].abs()
    shap_df["Feature"] = shap_df["Feature"].replace({
        "AREA NAME": "Patrol area",
        "Hour": "Hour of day",
        "DayOfWeek": "Day of week",
        "Month": "Month",
        "Year": "Year"
    })
    shap_df = (
        shap_df
        .sort_values("Absolute Contribution", ascending=False)
        .reset_index(drop=True)
    )

    return shap_df, predicted_class


# =========================================================
# DATA & MODEL LOADING
# =========================================================

@st.cache_data
def load_data():
    data = pd.read_csv(BASE_DIR / "Crime_Data_from_2020_to_Present.csv")
    data["DATE OCC"] = pd.to_datetime(data["DATE OCC"], errors="coerce")
    data["Year"] = data["DATE OCC"].dt.year
    data["Month"] = data["DATE OCC"].dt.month
    data["DayOfWeek"] = data["DATE OCC"].dt.dayofweek
    data["DayName"] = data["DATE OCC"].dt.day_name()
    data["TIME OCC"] = pd.to_numeric(data["TIME OCC"], errors="coerce")
    data["Hour"] = data["TIME OCC"] // 100
    data["Hour"] = pd.to_numeric(data["Hour"], errors="coerce")
    data["TimeOfDay"] = data["Hour"].apply(get_time_of_day)
    data["Season"] = data["Month"].apply(get_season)
    return data

df = load_data()

# MAPPINGS
area_mapping = (
    df[["AREA", "AREA NAME"]]
    .dropna()
    .drop_duplicates()
    .sort_values("AREA")
)
area_name_to_number = dict(zip(area_mapping["AREA NAME"], area_mapping["AREA"]))
area_number_to_name = dict(zip(area_mapping["AREA"], area_mapping["AREA NAME"]))

descent_mapping = {
    "Other Asian": "A",
    "Black": "B",
    "Chinese": "C",
    "Cambodian": "D",
    "Filipino": "F",
    "Guamanian": "G",
    "Hispanic / Latin / Mexican": "H",
    "American Indian / Alaskan Native": "I",
    "Japanese": "J",
    "Korean": "K",
    "Laotian": "L",
    "Other": "O",
    "Pacific Islander": "P",
    "Samoan": "S",
    "Hawaiian": "U",
    "Vietnamese": "V",
    "White": "W",
    "Unknown": "X",
    "Asian Indian": "Z"
}

sex_mapping = {
    "Female": "F",
    "Male": "M",
    "Unknown": "X"
}

@st.cache_resource
def load_models():
    classification_artifact = joblib.load(BASE_DIR / "best_crime_classification_model.pkl")
    risk_artifact = joblib.load(BASE_DIR / "crime_risk_model.pkl")
    classification_model = unwrap_pipeline(classification_artifact, ["model", "pipeline"])
    risk_model = unwrap_pipeline(risk_artifact, ["pipeline", "model"])
    classification_metadata = load_optional_artifact(
        "best_crime_classification_model_metadata.pkl", {}
    )
    risk_metadata = load_optional_artifact("crime_risk_model_metadata.pkl", {})
    if isinstance(classification_artifact, dict):
        classification_metadata = {**classification_artifact, **classification_metadata}
    if isinstance(risk_artifact, dict):
        risk_metadata = {**risk_artifact, **risk_metadata}
    return classification_model, risk_model, classification_metadata, risk_metadata

try:
    classification_model, risk_model, classification_metadata, risk_metadata = load_models()
except Exception as model_error:
    st.error("The saved model files are not compatible with the active project environment.")
    st.info(
        "Run the updated Jupyter notebook from top to bottom using the project .venv kernel. "
        "This will regenerate the classification and risk pipelines in the correct format."
    )
    with st.expander("Technical details"):
        st.exception(model_error)
    st.stop()
classification_features = get_pipeline_features(
    classification_model,
    classification_metadata,
    [
        "AREA NAME", "Vict Age", "Vict Sex", "Vict Descent", "Premis Desc",
        "Weapon Desc", "Hour", "TimeOfDay"
    ]
)
risk_features = get_pipeline_features(
    risk_model, risk_metadata, ["AREA NAME", "Hour", "DayOfWeek"]
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1.5rem;">
            <span style="font-size: 2rem;">🚨</span>
            <div>
                <h2 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: #f8fafc;">Crime Analytics</h2>
                <p style="margin: 0; font-size: 0.75rem; color: #94a3b8;">Intelligence & Prediction Platform</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase; margin-bottom: 0.5rem;'>Navigation</p>", unsafe_allow_html=True)
    
    page = st.radio(
        "Select Module",
        [
            "🏠 Dashboard",
            "📊 Historical Crime Analysis",
            "📄 Crime Information",
            "⚠️ Risk Prediction",
            "🗺️ Geographic Hotspots"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    with st.container(border=True):
        st.markdown("""
            <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.4;">
                <strong style="color: #38bdf8;">⚡ System Capabilities</strong><br>
                • Random Forest Risk Engine<br>
                • SHAP Explainability Layer<br>
                • Spatial Density Mapping<br>
                • Part 1 & 2 UCR Classifier
            </div>
        """, unsafe_allow_html=True)

    model_name = classification_metadata.get("model_name", "Random Forest")
    validation_year = risk_metadata.get("test_year", "latest complete year")
    st.markdown(f"""
        <div class="model-strip">
            <span class="model-chip">✓ Models loaded</span>
            <span class="model-chip">Classifier · {model_name}</span>
            <span class="model-chip">Risk validation · {validation_year}</span>
        </div>
    """, unsafe_allow_html=True)


# =========================================================
# MODULE 1: DASHBOARD
# =========================================================

if page == "🏠 Dashboard":
    st.markdown("""
        <div class="hero-container">
            <div class="eyebrow">Operational overview</div>
            <div class="hero-title">Crime Analytics Dashboard</div>
            <div class="hero-subtitle">High-level situational awareness: aggregate crime trends, area distributions, and temporal profiles across jurisdictions.</div>
        </div>
    """, unsafe_allow_html=True)

    # KPIS
    total_crimes = len(df)
    total_areas = df["AREA NAME"].nunique() if "AREA NAME" in df.columns else 0
    total_crime_types = df["Crm Cd Desc"].nunique() if "Crm Cd Desc" in df.columns else 0
    date_start = df["DATE OCC"].min()
    date_end = df["DATE OCC"].max()

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.metric("Total Crime Records", f"{total_crimes:,}", help="Total historical records in dataset")
    with kpi_col2:
        st.metric("Covered Areas", f"{total_areas:,}", help="Unique reporting patrol areas")
    with kpi_col3:
        st.metric("Crime Categories", f"{total_crime_types:,}", help="Unique crime code descriptions")
    with kpi_col4:
        period_str = f"{date_start.year} - {date_end.year}" if pd.notna(date_start) and pd.notna(date_end) else "N/A"
        st.metric("Data Timeline", period_str, help="Recorded timeframe")

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # FILTERS
    with st.container(border=True):
        st.markdown("<p style='font-size: 0.9rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.75rem;'>🔎 Global Dashboard Filters</p>", unsafe_allow_html=True)
        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            area_options = sorted(df["AREA NAME"].dropna().unique().tolist()) if "AREA NAME" in df.columns else []
            selected_area = st.selectbox("Area Jurisdiction", ["All Areas"] + area_options)

        with filter_col2:
            years = sorted(df["Year"].dropna().astype(int).unique().tolist())
            selected_year = st.selectbox("Incident Year", ["All Years"] + years)

        with filter_col3:
            crime_options = sorted(df["Crm Cd Desc"].dropna().unique().tolist()) if "Crm Cd Desc" in df.columns else []
            selected_crime = st.selectbox("Crime Classification", ["All Crime Types"] + crime_options)

    # APPLY FILTERS
    filtered_df = df.copy()
    if selected_area != "All Areas":
        filtered_df = filtered_df[filtered_df["AREA NAME"] == selected_area]
    if selected_year != "All Years":
        filtered_df = filtered_df[filtered_df["Year"] == selected_year]
    if selected_crime != "All Crime Types":
        filtered_df = filtered_df[filtered_df["Crm Cd Desc"] == selected_crime]

    st.caption(f"⚡ Showing **{len(filtered_df):,}** filtered incidents matching selection criteria.")

    # TREND CHART
    with st.container(border=True):
        st.markdown("### 📈 Crime Trend Over Time")
        if filtered_df.empty:
            st.warning("No data available for the selected filters.")
        else:
            if selected_year != "All Years":
                monthly_crimes = (
                    filtered_df
                    .groupby("Month")
                    .size()
                    .reindex(range(1, 13), fill_value=0)
                    .reset_index(name="Incident Count")
                )
                month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                monthly_crimes["Month"] = month_names
                
                fig_trend = px.line(
                    monthly_crimes, 
                    x="Month", 
                    y="Incident Count", 
                    markers=True,
                    line_shape="spline",
                    template="plotly_dark"
                )
                fig_trend.update_traces(line_color="#38bdf8", line_width=3, marker=dict(size=8, color="#818cf8"))
                fig_trend.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                yearly_crimes = (
                    filtered_df
                    .groupby("Year")
                    .size()
                    .reset_index(name="Incident Count")
                    .sort_values("Year")
                )
                fig_trend = px.area(
                    yearly_crimes, 
                    x="Year", 
                    y="Incident Count", 
                    markers=True,
                    template="plotly_dark"
                )
                fig_trend.update_traces(line_color="#38bdf8", fillcolor="rgba(56, 189, 248, 0.15)")
                fig_trend.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig_trend, use_container_width=True)

    # TWO COLUMNS: TOP AREAS & TOP CRIMES
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        with st.container(border=True):
            st.markdown("### 📍 Top Incident Areas")
            if "AREA NAME" in filtered_df.columns:
                area_counts = filtered_df["AREA NAME"].value_counts().head(10).reset_index()
                area_counts.columns = ["Area", "Incidents"]
                fig_area = px.bar(
                    area_counts, 
                    x="Incidents", 
                    y="Area", 
                    orientation="h",
                    template="plotly_dark",
                    color="Incidents",
                    color_continuous_scale="Viridis"
                )
                fig_area.update_layout(
                    yaxis=dict(categoryorder="total ascending"),
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_area, use_container_width=True)

    with col_chart2:
        with st.container(border=True):
            st.markdown("### 🔎 Most Frequent Crime Types")
            if "Crm Cd Desc" in filtered_df.columns:
                crime_counts = filtered_df["Crm Cd Desc"].value_counts().head(10).reset_index()
                crime_counts.columns = ["Crime Type", "Incidents"]
                fig_crime = px.bar(
                    crime_counts, 
                    x="Incidents", 
                    y="Crime Type", 
                    orientation="h",
                    template="plotly_dark",
                    color="Incidents",
                    color_continuous_scale="Plasma"
                )
                fig_crime.update_layout(
                    yaxis=dict(categoryorder="total ascending"),
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_crime, use_container_width=True)

    # TIME OF DAY ANALYSIS
    with st.container(border=True):
        st.markdown("### 🕐 Distribution by Time of Day")
        time_counts = (
            filtered_df["TimeOfDay"]
            .value_counts()
            .reindex(["Morning", "Afternoon", "Evening", "Night", "Unknown"], fill_value=0)
            .reset_index()
        )
        time_counts.columns = ["Time of Day", "Incidents"]
        fig_tod = px.bar(
            time_counts,
            x="Time of Day",
            y="Incidents",
            template="plotly_dark",
            color="Incidents",
            color_continuous_scale="Sunset"
        )
        fig_tod.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_tod, use_container_width=True)

    # RECENT RECORDS TABLE
    with st.container(border=True):
        st.markdown("### 📋 Sample Crime Log Entries")
        display_columns = [
            "DATE OCC", "AREA NAME", "Crm Cd Desc", "Vict Age",
            "Vict Sex", "Weapon Desc", "TimeOfDay"
        ]
        available_columns = [col for col in display_columns if col in filtered_df.columns]
        st.dataframe(
            filtered_df[available_columns].head(100),
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# MODULE 2: HISTORICAL CRIME ANALYSIS
# =========================================================

elif page == "📊 Historical Crime Analysis":
    st.markdown("""
        <div class="hero-container">
            <div class="eyebrow">Pattern intelligence</div>
            <div class="hero-title">Historical Crime Analytics</div>
            <div class="hero-subtitle">Comprehensive demographic breakdown, seasonal periodicity, weapon utilization, and longitudinal patterns.</div>
        </div>
    """, unsafe_allow_html=True)

    # FILTERS
    with st.container(border=True):
        st.markdown("<p style='font-size: 0.9rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.75rem;'>🔎 Filter Historical Records</p>", unsafe_allow_html=True)
        hist_col1, hist_col2 = st.columns(2)
        with hist_col1:
            hist_area_options = sorted(df["AREA NAME"].dropna().unique().tolist())
            hist_area = st.selectbox("Jurisdiction / Area", ["All Areas"] + hist_area_options, key="historical_area")
        with hist_col2:
            hist_year_options = sorted(df["Year"].dropna().astype(int).unique().tolist())
            hist_year = st.selectbox("Selected Year", ["All Years"] + hist_year_options, key="historical_year")

    # APPLY FILTERS
    historical_df = df.copy()
    if hist_area != "All Areas":
        historical_df = historical_df[historical_df["AREA NAME"] == hist_area]
    if hist_year != "All Years":
        historical_df = historical_df[historical_df["Year"] == hist_year]

    if historical_df.empty:
        st.warning("No crime records are available for the selected filters.")
    else:
        # KPIS
        total_records = len(historical_df)
        unique_crimes = historical_df["Crm Cd Desc"].nunique()
        unique_areas = historical_df["AREA NAME"].nunique()
        avg_victim_age = historical_df["Vict Age"].mean()

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Total Records", f"{total_records:,}")
        with kpi2:
            st.metric("Unique Crime Types", f"{unique_crimes:,}")
        with kpi3:
            st.metric("Patrol Areas", f"{unique_areas:,}")
        with kpi4:
            st.metric("Avg Victim Age", f"{avg_victim_age:.1f}" if pd.notna(avg_victim_age) else "N/A")

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        # YEARLY TREND
        with st.container(border=True):
            st.markdown("### 📈 Yearly Incident Trajectory")
            yearly_data = (
                historical_df.groupby("Year")
                .size()
                .reset_index(name="Crime Count")
                .sort_values("Year")
            )
            if len(yearly_data) > 1:
                fig_year = px.line(
                    yearly_data,
                    x="Year",
                    y="Crime Count",
                    markers=True,
                    template="plotly_dark",
                    title="Historical Total Incidents by Year"
                )
                fig_year.update_traces(line_color="#38bdf8", line_width=3, marker=dict(size=8, color="#a855f7"))
                fig_year.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    hovermode="x unified",
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_year, use_container_width=True)
            else:
                st.info("💡 Select 'All Years' in the filter above to view the complete multi-year trajectory.")

        # MONTHLY & DAY OF WEEK
        col_m, col_d = st.columns(2)
        
        with col_m:
            with st.container(border=True):
                st.markdown("### 📅 Monthly Seasonality")
                monthly_data = (
                    historical_df.groupby("Month")
                    .size()
                    .reindex(range(1, 13), fill_value=0)
                    .reset_index()
                )
                monthly_data.columns = ["Month", "Crime Count"]
                month_names = {
                    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
                    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
                    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
                }
                monthly_data["Month Name"] = monthly_data["Month"].map(month_names)
                fig_month = px.bar(
                    monthly_data,
                    x="Month Name",
                    y="Crime Count",
                    template="plotly_dark",
                    color="Crime Count",
                    color_continuous_scale="Tealgrn"
                )
                fig_month.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False,
                    margin=dict(l=10, r=10, t=20, b=10)
                )
                st.plotly_chart(fig_month, use_container_width=True)

        with col_d:
            with st.container(border=True):
                st.markdown("### 📆 Incidents by Day of Week")
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                day_data = (
                    historical_df["DayName"]
                    .value_counts()
                    .reindex(day_order, fill_value=0)
                    .reset_index()
                )
                day_data.columns = ["Day", "Crime Count"]
                fig_day = px.bar(
                    day_data,
                    x="Day",
                    y="Crime Count",
                    template="plotly_dark",
                    color="Crime Count",
                    color_continuous_scale="Purp"
                )
                fig_day.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False,
                    margin=dict(l=10, r=10, t=20, b=10)
                )
                st.plotly_chart(fig_day, use_container_width=True)

        # TIME OF DAY & SEASON
        time_col, season_col = st.columns(2)
        with time_col:
            with st.container(border=True):
                st.markdown("### 🕐 Time of Day Breakdown")
                time_order = ["Morning", "Afternoon", "Evening", "Night", "Unknown"]
                time_data = (
                    historical_df["TimeOfDay"]
                    .value_counts()
                    .reindex(time_order, fill_value=0)
                    .reset_index()
                )
                time_data.columns = ["Time of Day", "Crime Count"]
                fig_time = px.bar(
                    time_data,
                    x="Time of Day",
                    y="Crime Count",
                    template="plotly_dark",
                    color="Crime Count",
                    color_continuous_scale="Sunsetdark"
                )
                fig_time.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False,
                    margin=dict(l=10, r=10, t=20, b=10)
                )
                st.plotly_chart(fig_time, use_container_width=True)

        with season_col:
            with st.container(border=True):
                st.markdown("### 🌦️ Seasonal Concentration")
                season_order = ["Winter", "Spring", "Summer", "Autumn", "Unknown"]
                season_data = (
                    historical_df["Season"]
                    .value_counts()
                    .reindex(season_order, fill_value=0)
                    .reset_index()
                )
                season_data.columns = ["Season", "Crime Count"]
                fig_season = px.bar(
                    season_data,
                    x="Season",
                    y="Crime Count",
                    template="plotly_dark",
                    color="Crime Count",
                    color_continuous_scale="Burg"
                )
                fig_season.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False,
                    margin=dict(l=10, r=10, t=20, b=10)
                )
                st.plotly_chart(fig_season, use_container_width=True)

        # DEMOGRAPHICS & WEAPONS
        st.markdown("### 👤 Demographic & Modus Operandi Profiles")
        dem_col1, dem_col2 = st.columns(2)

        with dem_col1:
            with st.container(border=True):
                st.markdown("#### Victim Age Distribution")
                age_data = historical_df.copy()
                age_data["Victim Age"] = pd.to_numeric(age_data["Vict Age"], errors="coerce")
                age_data = age_data[(age_data["Victim Age"] > 0) & (age_data["Victim Age"] <= 100)]
                if not age_data.empty:
                    fig_age = px.histogram(
                        age_data,
                        x="Victim Age",
                        nbins=20,
                        template="plotly_dark",
                        color_discrete_sequence=["#38bdf8"]
                    )
                    fig_age.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=10, r=10, t=20, b=10)
                    )
                    st.plotly_chart(fig_age, use_container_width=True)
                else:
                    st.info("No valid victim age data available.")

        with dem_col2:
            with st.container(border=True):
                st.markdown("#### Victim Sex Distribution")
                sex_data = (
                    historical_df["Vict Sex"]
                    .fillna("Unknown")
                    .replace("", "Unknown")
                    .value_counts()
                    .reset_index()
                )
                sex_data.columns = ["Victim Sex", "Crime Count"]
                fig_sex = px.pie(
                    sex_data,
                    names="Victim Sex",
                    values="Crime Count",
                    template="plotly_dark",
                    hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_sex.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=20, b=10)
                )
                st.plotly_chart(fig_sex, use_container_width=True)

        # WEAPONS
        with st.container(border=True):
            st.markdown("### 🔪 Top Weapon Categories Utilized")
            weapon_data = (
                historical_df["Weapon Desc"]
                .fillna("No Weapon / Unknown")
                .replace("", "No Weapon / Unknown")
                .value_counts()
                .head(10)
                .reset_index()
            )
            weapon_data.columns = ["Weapon", "Crime Count"]
            fig_weapon = px.bar(
                weapon_data,
                x="Crime Count",
                y="Weapon",
                orientation="h",
                template="plotly_dark",
                color="Crime Count",
                color_continuous_scale="Reds"
            )
            fig_weapon.update_layout(
                yaxis={"categoryorder": "total ascending"},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_weapon, use_container_width=True)

        # DATA TABLE
        with st.container(border=True):
            st.markdown("### 📋 Historical Records Archive")
            historical_columns = [
                "DATE OCC", "AREA NAME", "Crm Cd Desc", "Vict Age",
                "Vict Sex", "Weapon Desc", "TimeOfDay", "Season"
            ]
            historical_available = [col for col in historical_columns if col in historical_df.columns]
            st.dataframe(
                historical_df[historical_available].head(200),
                use_container_width=True,
                hide_index=True
            )


# =========================================================
# MODULE 3: CRIME CLASSIFICATION
# =========================================================

elif page == "📄 Crime Information":
    st.markdown("""
        <div class="hero-container">
            <div class="eyebrow">Incident classification</div>
            <div class="hero-title">Crime Classification Engine</div>
            <div class="hero-subtitle">Classify prospective or reported incidents into FBI Uniform Crime Reporting (UCR) Part 1 or Part 2 categories using machine learning.</div>
        </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<p style='font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-bottom: 1rem;'>📝 Incident Attributes Input</p>", unsafe_allow_html=True)
        
        # ROW 1: Location & Victim Profile
        col1, col2, col3 = st.columns(3)
        with col1:
            area_names = sorted(df["AREA NAME"].dropna().astype(str).unique().tolist())
            input_area_name = st.selectbox("Area Jurisdiction", area_names, key="classification_area_name")
            input_area = area_name_to_number[input_area_name]
            st.caption(f"Assigned Area ID: `{input_area}`")

        with col2:
            input_victim_age = st.number_input("Victim Age", min_value=0, max_value=100, value=30, step=1, key="classification_age")

        with col3:
            sex_options = list(sex_mapping.keys())
            input_victim_sex_label = st.selectbox("Victim Sex", sex_options, key="classification_sex")
            input_victim_sex = sex_mapping[input_victim_sex_label]

        # ROW 2: Demographics & Context
        col4, col5, col6 = st.columns(3)
        with col4:
            descent_options = list(descent_mapping.keys())
            input_victim_descent_label = st.selectbox("Victim Descent", descent_options, key="classification_descent")
            input_victim_descent = descent_mapping[input_victim_descent_label]

        with col5:
            premise_options = sorted(df["Premis Desc"].dropna().astype(str).unique().tolist())
            input_premise = st.selectbox("Premise Type", premise_options, key="classification_premise")

        with col6:
            weapon_options = sorted(df["Weapon Desc"].fillna("Unknown").astype(str).unique().tolist())
            input_weapon = st.selectbox("Weapon Involved", weapon_options, key="classification_weapon")

        # ROW 3: Temporal Settings
        col7, col8, col9 = st.columns(3)
        with col7:
            input_hour = st.slider("Hour of Occurrence (24h)", min_value=0, max_value=23, value=12, step=1, key="classification_hour")
        with col8:
            input_month = st.slider("Month of Occurrence", min_value=1, max_value=12, value=6, step=1, key="classification_month")
        with col9:
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            input_day_name = st.selectbox("Day of Week", day_names, key="classification_day")
            input_day_of_week = day_names.index(input_day_name)

        # DERIVED BADGES
        input_is_weekend = 1 if input_day_of_week >= 5 else 0
        input_time_of_day = get_time_of_day(input_hour)
        season = get_season(input_month)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown(f"""
            <div style="display: flex; gap: 10px; flex-wrap: wrap; background: rgba(15, 23, 42, 0.5); padding: 0.85rem 1.2rem; border-radius: 10px; border: 1px solid rgba(255,255,255,0.06);">
                <span style="color: #94a3b8; font-size: 0.85rem; font-weight: 600;">Derived Context:</span>
                <span class="result-badge badge-info">🕒 {input_time_of_day}</span>
                <span class="result-badge badge-info">🌿 {season}</span>
                <span class="result-badge {'badge-warning' if input_is_weekend else 'badge-info'}">{'📅 Weekend' if input_is_weekend else '💼 Weekday'}</span>
            </div>
        """, unsafe_allow_html=True)

    with st.expander("ℹ️ What do UCR Part 1 and Part 2 categories indicate?"):
        st.markdown("""
            **Part 1 Crimes (Major Offenses):**
            Major violent and property crimes prioritized for national statistical benchmarks, including Homicide, Rape, Robbery, Aggravated Assault, Burglary, Larceny-Theft, Motor Vehicle Theft, and Arson.

            **Part 2 Crimes (Secondary Offenses):**
            All other offenses recorded in the uniform reporting standard (e.g., vandalism, fraud, simple assault, narcotics violations, disorderly conduct).
        """)

    predict_button = st.button("🚨 Classify Crime Incident", use_container_width=True, type="primary", key="classification_predict_button")

    if predict_button:
        try:
            classification_values = {
                "AREA": input_area,
                "AREA NAME": input_area_name,
                "Vict Age": input_victim_age,
                "Vict Sex": input_victim_sex,
                "Vict Descent": input_victim_descent,
                "Premis Desc": input_premise,
                "Weapon Desc": input_weapon,
                "Hour": input_hour,
                "Month": input_month,
                "DayOfWeek": input_day_of_week,
                "IsWeekend": input_is_weekend,
                "TimeOfDay": input_time_of_day,
                "Season": season
            }
            input_data = build_model_input(classification_values, classification_features)
            prediction = classification_model.predict(input_data)[0]
            classification_labels = {0: "Part 1 Crime", 1: "Part 2 Crime"}
            predicted_label = classification_labels.get(int(prediction), f"Unknown Class ({prediction})")

            st.markdown("### 🎯 Classification Result")
            is_part1 = predicted_label == "Part 1 Crime"

            st.markdown(f"""
                <div class="prediction-box {'high' if is_part1 else 'medium'}">
                    <div style="font-size: 0.9rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; color: {'#f87171' if is_part1 else '#fbbf24'};">
                        Target Prediction Outcome
                    </div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #ffffff; margin: 0.3rem 0;">
                        {predicted_label}
                    </div>
                    <div style="font-size: 0.95rem; color: #cbd5e1; line-height: 1.4;">
                        {'Classified as a major index offense (e.g., robbery, burglary, aggravated assault, grand theft).' if is_part1 else 'Classified under secondary UCR offense groups (e.g., simple assault, vandalism, non-index crimes).'}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            if hasattr(classification_model, "predict_proba"):
                probabilities = classification_model.predict_proba(input_data)[0]
                classes = classification_model.classes_

                readable_classes = [
                    classification_labels.get(int(cls), f"Unknown Class ({cls})")
                    for cls in classes
                ]
                probability_df = pd.DataFrame({
                    "Crime Classification": readable_classes,
                    "Probability": probabilities
                })
                probability_df["Probability (%)"] = (probability_df["Probability"] * 100).round(2)
                probability_df["Meaning"] = probability_df["Crime Classification"].map({
                    "Part 1 Crime": "Major UCR offense group",
                    "Part 2 Crime": "Other UCR offense group"
                })

                with st.container(border=True):
                    st.markdown("#### 📊 Confidence Spectrum")
                    prob_col1, prob_col2 = st.columns([1, 1])
                    with prob_col1:
                        fig_conf = px.bar(
                            probability_df,
                            x="Crime Classification",
                            y="Probability (%)",
                            template="plotly_dark",
                            color="Crime Classification",
                            color_discrete_map={"Part 1 Crime": "#ef4444", "Part 2 Crime": "#3b82f6"}
                        )
                        fig_conf.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            showlegend=False,
                            margin=dict(l=10, r=10, t=20, b=10)
                        )
                        st.plotly_chart(fig_conf, use_container_width=True)

                    with prob_col2:
                        st.dataframe(
                            probability_df[["Crime Classification", "Meaning", "Probability (%)"]],
                            use_container_width=True,
                            hide_index=True
                        )
                        highest_prob = probability_df["Probability (%)"].max()
                        st.metric("Classification Confidence", f"{highest_prob:.2f}%")

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)


# =========================================================
# MODULE 4: RISK PREDICTION & SHAP
# =========================================================

elif page == "⚠️ Risk Prediction":
    st.markdown("""
        <div class="hero-container">
            <div class="eyebrow">Retrospective area-time model</div>
            <div class="hero-title">Historical Area-Time Risk Estimator</div>
            <div class="hero-subtitle">Estimate a historical LOW, MEDIUM, or HIGH risk category for an area-time combination, with transparent SHAP feature attribution.</div>
        </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<p style='font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-bottom: 1rem;'>📍 Location & Temporal Target</p>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            risk_area_names = sorted(df["AREA NAME"].dropna().astype(str).unique().tolist())
            risk_area_name = st.selectbox("Patrol Area Name", risk_area_names, key="risk_area_name")
            risk_area = area_name_to_number[risk_area_name]
            st.caption(f"Area Code Identifier: `{risk_area}`")

        with col2:
            risk_hour = st.slider("Hour of Day (0 - 23)", min_value=0, max_value=23, value=12, step=1, key="risk_hour")
            risk_day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            risk_day_name = st.selectbox("Day of Week", risk_day_names, key="risk_day")
            risk_day_of_week = risk_day_names.index(risk_day_name)

    risk_predict_button = st.button("⚠️ Evaluate Crime Risk Level", use_container_width=True, type="primary", key="risk_predict_button")

    if risk_predict_button:
        try:
            latest_date = df["DATE OCC"].max()
            risk_values = {
                "AREA": risk_area,
                "AREA NAME": risk_area_name,
                "Hour": risk_hour,
                "DayOfWeek": risk_day_of_week,
                # Backward compatibility for a previously saved chronological artifact.
                "Year": int(risk_metadata.get("test_year", latest_date.year)),
                "Month": int(latest_date.month)
            }
            risk_input = build_model_input(risk_values, risk_features)
            risk_prediction = risk_model.predict(risk_input)[0]

            if hasattr(risk_model, "predict_proba"):
                risk_probabilities = risk_model.predict_proba(risk_input)[0]
                risk_classes = risk_model.classes_
                risk_probability_df = pd.DataFrame({
                    "Risk Level": risk_classes,
                    "Probability (%)": (risk_probabilities * 100).round(2)
                }).sort_values("Probability (%)", ascending=False).reset_index(drop=True)
            else:
                risk_probability_df = pd.DataFrame({
                    "Risk Level": [risk_prediction],
                    "Probability (%)": [100.0]
                })

            st.markdown("### 🎯 Historical Risk Estimate")
            pred_class_str = str(risk_prediction).upper()
            box_type = "high" if "HIGH" in pred_class_str else ("medium" if "MED" in pred_class_str else "low")
            badge_color = "#f87171" if box_type == "high" else ("#fbbf24" if box_type == "medium" else "#34d399")

            st.markdown(f"""
                <div class="prediction-box {box_type}">
                    <div style="font-size: 0.9rem; font-weight: 700; text-transform: uppercase; color: {badge_color};">
                        Estimated Historical Risk Category
                    </div>
                    <div style="font-size: 2.3rem; font-weight: 800; color: #ffffff; margin: 0.2rem 0;">
                        {risk_prediction} RISK
                    </div>
                    <div style="font-size: 0.95rem; color: #cbd5e1;">
                        Evaluated for <strong>{risk_area_name}</strong> at <strong>{risk_hour:02d}:00 hours</strong> on <strong>{risk_day_name}</strong>.
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.caption(
                "This category summarizes patterns learned from historical area-time counts. "
                "It does not predict that a crime will occur."
            )

            # PROBABILITY BREAKDOWN
            with st.container(border=True):
                st.markdown("#### 📊 Prediction Confidence Breakdown")
                c_col1, c_col2 = st.columns([1, 1])
                with c_col1:
                    fig_r_prob = px.bar(
                        risk_probability_df,
                        x="Risk Level",
                        y="Probability (%)",
                        template="plotly_dark",
                        color="Risk Level",
                        color_discrete_map={"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#10b981"}
                    )
                    fig_r_prob.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        showlegend=False,
                        margin=dict(l=10, r=10, t=10, b=10)
                    )
                    st.plotly_chart(fig_r_prob, use_container_width=True)

                with c_col2:
                    st.dataframe(risk_probability_df, use_container_width=True, hide_index=True)
                    top_prob = risk_probability_df.iloc[0]["Probability (%)"]
                    st.metric("Top Classification Confidence", f"{top_prob:.2f}%")

            # SHAP EXPLAINABILITY
            with st.container(border=True):
                st.markdown("### 🔍 Model Explainability (SHAP Analysis)")
                st.write("Understand feature contributions determining this specific risk classification.")

                try:
                    shap_df, shap_predicted_class = get_risk_shap_explanation(
                        risk_model, risk_input, risk_features
                    )
                    display_shap_df = shap_df.copy()
                    display_shap_df["Contribution"] = display_shap_df["Contribution"].round(4)
                    display_shap_df["Direction"] = display_shap_df["Contribution"].apply(
                        lambda x: "↑ Toward predicted risk" if x > 0 else "↓ Away from predicted risk"
                    )

                    shap_col1, shap_col2 = st.columns([1, 1])

                    with shap_col1:
                        fig_shap = px.bar(
                            display_shap_df,
                            x="Contribution",
                            y="Feature",
                            orientation="h",
                            template="plotly_dark",
                            color="Contribution",
                            color_continuous_scale="RdBu_r"
                        )
                        fig_shap.update_layout(
                            yaxis={"categoryorder": "total ascending"},
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            coloraxis_showscale=False,
                            margin=dict(l=10, r=10, t=10, b=10)
                        )
                        st.plotly_chart(fig_shap, use_container_width=True)

                    with shap_col2:
                        st.dataframe(
                            display_shap_df[["Feature", "Contribution", "Direction"]],
                            use_container_width=True,
                            hide_index=True
                        )

                    strongest_feature = shap_df.iloc[0]["Feature"]
                    strongest_contribution = shap_df.iloc[0]["Contribution"]

                    if strongest_contribution > 0:
                        st.info(f"💡 **{strongest_feature}** had the strongest positive contribution, pushing the prediction towards **{shap_predicted_class}** risk.")
                    elif strongest_contribution < 0:
                        st.info(f"💡 **{strongest_feature}** had the strongest negative contribution, pulling the prediction away from **{shap_predicted_class}** risk.")
                    else:
                        st.info("The calculated feature contributions were near zero for this input.")

                except Exception as shap_error:
                    st.warning("SHAP explanation could not be generated for this prediction.")
                    st.exception(shap_error)

        except Exception as e:
            st.error("Risk prediction failed.")
            st.exception(e)


# =========================================================
# MODULE 5: GEOGRAPHIC HOTSPOTS
# =========================================================

elif page == "🗺️ Geographic Hotspots":
    st.markdown("""
        <div class="hero-container">
            <div class="eyebrow">Spatial intelligence</div>
            <div class="hero-title">Geographic Crime Hotspots</div>
            <div class="hero-subtitle">Interactive spatial density visualization powered by Folium heatmaps to discover high-density crime clusters.</div>
        </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<p style='font-size: 0.9rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.75rem;'>🔎 Spatial Heatmap Filters</p>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            hotspot_area = st.selectbox(
                "Area Jurisdiction",
                ["All"] + sorted(df["AREA NAME"].dropna().unique().tolist()),
                key="hotspot_area"
            )
        with col2:
            hotspot_year = st.selectbox(
                "Filter Year",
                ["All"] + sorted(df["Year"].dropna().astype(int).unique().tolist()),
                key="hotspot_year"
            )
        with col3:
            hotspot_crime = st.selectbox(
                "Crime Type",
                ["All"] + sorted(df["Crm Cd Desc"].dropna().unique().tolist()),
                key="hotspot_crime"
            )

    # APPLY FILTERS
    hotspot_df = df.copy()
    if hotspot_area != "All":
        hotspot_df = hotspot_df[hotspot_df["AREA NAME"] == hotspot_area]
    if hotspot_year != "All":
        hotspot_df = hotspot_df[hotspot_df["Year"] == int(hotspot_year)]
    if hotspot_crime != "All":
        hotspot_df = hotspot_df[hotspot_df["Crm Cd Desc"] == hotspot_crime]

    hotspot_df["LAT"] = pd.to_numeric(hotspot_df["LAT"], errors="coerce")
    hotspot_df["LON"] = pd.to_numeric(hotspot_df["LON"], errors="coerce")
    hotspot_df = hotspot_df.dropna(subset=["LAT", "LON"])
    hotspot_df = hotspot_df[(hotspot_df["LAT"] != 0) & (hotspot_df["LON"] != 0)]

    st.markdown(f"""
        <div style="background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 10px; padding: 0.75rem 1.2rem; margin: 1rem 0; color: #38bdf8; font-size: 0.9rem;">
            📍 Rendering <strong>{len(hotspot_df):,}</strong> validated geo-tagged crime incidents.
        </div>
    """, unsafe_allow_html=True)

    if len(hotspot_df) == 0:
        st.warning("No geographic crime records are available for the selected filters.")
    else:
        with st.container(border=True):
            st.markdown("### 🔥 Spatial Incident Density Map")
            center_lat = hotspot_df["LAT"].mean()
            center_lon = hotspot_df["LON"].mean()

            crime_map = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=11,
                tiles="CartoDB dark_matter"
            )

            map_limit = 50_000
            map_df = hotspot_df.sample(
                n=min(map_limit, len(hotspot_df)), random_state=42
            )
            heat_data = map_df[["LAT", "LON"]].values.tolist()
            HeatMap(
                heat_data,
                radius=11,
                blur=16,
                min_opacity=0.35
            ).add_to(crime_map)

            st_folium(
                crime_map,
                width=None,
                height=600,
                returned_objects=[]
            )
            if len(hotspot_df) > map_limit:
                st.caption(
                    f"Map rendered from a reproducible sample of {map_limit:,} incidents "
                    f"for responsive interaction; rankings use all {len(hotspot_df):,} filtered records."
                )

        with st.container(border=True):
            st.markdown("### 📊 Concentration Ranking by Area")
            area_counts = (
                hotspot_df["AREA NAME"]
                .value_counts()
                .reset_index()
            )
            area_counts.columns = ["Area", "Crime Count"]

            rank_col1, rank_col2 = st.columns([1, 1])
            with rank_col1:
                fig_hot_bar = px.bar(
                    area_counts.head(10),
                    x="Crime Count",
                    y="Area",
                    orientation="h",
                    template="plotly_dark",
                    color="Crime Count",
                    color_continuous_scale="Oranges"
                )
                fig_hot_bar.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False,
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig_hot_bar, use_container_width=True)

            with rank_col2:
                st.dataframe(
                    area_counts.head(10),
                    use_container_width=True,
                    hide_index=True
                )
