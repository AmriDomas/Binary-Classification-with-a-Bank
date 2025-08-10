# app.py (fixed)
import streamlit as st

# ===== MUST be the first Streamlit command =====
st.set_page_config(page_title="Bank Marketing App", layout="wide")
# =================================================

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import requests
import os
import tempfile

# ===== Helper untuk download & cache file dari URL =====
@st.cache_data(show_spinner=False)
def download_file(url, filename=None):
    if filename is None:
        filename = url.split("/")[-1]
    cache_path = os.path.join(tempfile.gettempdir(), filename)
    if not os.path.exists(cache_path):
        r = requests.get(url)
        r.raise_for_status()
        with open(cache_path, "wb") as f:
            f.write(r.content)
    return cache_path

# ===== Load model dari Hugging Face =====
@st.cache_resource(show_spinner=True)
def load_model_from_hf():
    hf_model_url = "https://huggingface.co/11amri/xgboostbank/main/bank_model.pkl"  # ganti dengan link file model kamu di HF
    model_path = download_file(hf_model_url)
    return joblib.load(model_path)

# ===== Load dataset dari Hugging Face =====
@st.cache_data(show_spinner=True)
def load_data_from_hf():
    hf_data_url = "https://huggingface.co/datasets/11amri/banktrain/main/train.csv"  # ganti dengan link file csv kamu di HF
    csv_path = download_file(hf_data_url)
    return pd.read_csv(csv_path)

# Load data dan model
df_train = load_data_from_hf()
model = load_model_from_hf()

# Kolom kategorikal sesuai training
cat_cols = [
    'job', 'marital', 'education', 'default', 'housing', 'loan',
    'contact', 'month', 'poutcome'
]

# ====== Prepare data & encoders ======
df_train = df_train.copy()

# safety: pastikan kolom ada
missing = [c for c in cat_cols if c not in df_train.columns]
if missing:
    raise ValueError(f"Missing categorical columns in train.csv: {missing}")

encoders = {col: LabelEncoder().fit(df_train[col]) for col in cat_cols}

# ====== Sidebar & Navigation ======
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Select Menu", ["Analysis", "Prediction"])

# ====== ANALYSIS ======
if menu == "Analysis":
    st.title("📊 Data Analysis - Bank Marketing")
    df = df_train.copy()
    
    # CSS untuk tab biar rata
    st.markdown("""
    <style>
    /* Tab container rata */
    div[data-baseweb="tab-list"] {
        justify-content: space-between !important;
    }

    /* Ukuran font tab 10px */
    div[data-baseweb="tab"] > button {
        font-size: 20px !important;
        text-align: center !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 4px;
        padding: 6px 10px !important;
        line-height: 1.2 !important;
    }

    /* Tab aktif */
    div[data-baseweb="tab"][aria-selected="true"] > button {
        background-color: #1976d2 !important;
        color: white !important;
        border-radius: 6px 6px 0 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Tab setup
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Preview", "📈 Stats", "🔥 Correlation", "🎨 Custom Plot", "📈 Time Series By Month"
    ])

    def get_dynamic_palette(n):
            """Generate palette: if n>3 use darkening gradient, else fixed Set2."""
            if n <= 3:
                return sns.color_palette("Set2", n)
            else:
                # gradasi dari terang ke gelap (Blues)
                return sns.color_palette("Blues", n)

    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(15), use_container_width=True)


    def plot_categorical(df, col):
        num_cat = df[col].nunique(dropna=False)
        palette = get_dynamic_palette(num_cat)

        fig, ax = plt.subplots(figsize=(6, 3))
        order = df[col].value_counts().index
        sns.countplot(x=col, data=df, order=order, palette=palette, ax=ax)

        # Title & ticks lebih proporsional
        ax.set_title(f"Distribusi {col}", fontsize=8, weight='bold')
        ax.set_xlabel(col, fontsize=6)  # benerin param
        ax.set_ylabel("Count", fontsize=6)
        ax.tick_params(axis='x', rotation=45, labelsize=5)
        ax.tick_params(axis='y', labelsize=5)

        # Label bar lebih kecil, posisinya di edge
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', fontsize=4, label_type='edge', padding=1)

        sns.despine()  # buang border luar
        st.pyplot(fig)


    with tab2:

        st.markdown("""
        <style>
        [data-testid="stDataFrame"] {
            width: 100% !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.subheader("Basic Statistics")
        st.write(df.describe(include='all'), use_container_width=True)

        st.subheader("📊 Distribution Category")
        col_kategori = st.selectbox("Select the category column", df.select_dtypes(include='object').columns)

        fig_width = max(6, len(df[col_kategori].unique()) * 0.6)  # dinamis
        fig, ax = plt.subplots(figsize=(fig_width, 4))

        order = df[col_kategori].value_counts().index
        num_cat = len(order)
        palette = get_dynamic_palette(num_cat)

        sns.countplot(
            x=col_kategori, 
            data=df, 
            order=order, 
            palette=palette, 
            width=0.6,  # lebih ramping
            ax=ax
        )

        ax.set_title(f"Distribution {col_kategori}", fontsize=10, weight='bold')
        ax.set_xlabel(col_kategori, fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)

        for container in ax.containers:
            ax.bar_label(container, fmt='%d', fontsize=6, label_type='edge', padding=1)

        sns.despine()
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("📊 Distribusi Kategori (Pie Chart)")

        # Gunakan col_kategori yang sudah dipilih di bar chart biar sinkron
        val_counts = df[col_kategori].value_counts()

        fig, ax = plt.subplots(figsize=(6, 4))

        # Pie chart dengan persentase
        wedges, texts, autotexts = ax.pie(
            val_counts,
            labels=None,  # label di pie dihilangin biar bersih
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("Blues", len(val_counts))
        )

        # Hitung persentase
        percentages = val_counts / val_counts.sum() * 100

        ax.legend(
            wedges,
            [f"{cat} ({p:.1f}%)" for cat, p in zip(val_counts.index, percentages)],
            title=col_kategori,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=5,
            title_fontsize=8
        )

        plt.tight_layout()
        st.pyplot(fig)


    with tab3:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        corr = df.drop(columns=["id"], errors='ignore').corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap="Spectral", ax=ax)
        st.pyplot(fig)

    with tab4:
        st.subheader("📊 Custom Plot (Auto Bar / Scatter / Box)")
        all_cols = df.columns.tolist()
        col_x = st.selectbox("Select Column X", all_cols, index=0)
        col_y = st.selectbox("Select Column Y", all_cols, index=1)

        fig, ax = plt.subplots(figsize=(8, 4))
        x_is_cat = df[col_x].dtype == 'object'
        y_is_cat = df[col_y].dtype == 'object'

        if x_is_cat and y_is_cat:
            num_cat = df[col_y].nunique(dropna=False)
            palette = get_dynamic_palette(num_cat)
            sns.countplot(x=col_x, hue=col_y, data=df, palette=palette, ax=ax)
            ax.set_title(f"Bar Chart: {col_x} vs {col_y}", fontsize=8, weight='bold')
            ax.set_xlabel(col_x, fontsize=6)
            ax.set_ylabel("Count", fontsize=6)
            ax.tick_params(axis='x', rotation=45, labelsize=5)
            ax.tick_params(axis='y', labelsize=5)
            for container in ax.containers:
                ax.bar_label(container, fmt='%d', fontsize=4, label_type='edge', padding=1)

        elif not x_is_cat and not y_is_cat:
            sns.scatterplot(x=col_x, y=col_y, data=df, color="royalblue", ax=ax)
            ax.set_title(f"Scatter Plot: {col_x} vs {col_y}", fontsize=8, weight='bold')
            ax.set_xlabel(col_x, fontsize=6)
            ax.set_ylabel(col_y, fontsize=6)
            ax.tick_params(axis='both', labelsize=5)

        else:
            if x_is_cat:
                num_cat = df[col_x].nunique(dropna=False)
                palette = get_dynamic_palette(num_cat)
                sns.boxplot(x=col_x, y=col_y, data=df, palette=palette, ax=ax)
                ax.set_xlabel(col_x, fontsize=6)
                ax.set_ylabel(col_y, fontsize=6)
            else:
                num_cat = df[col_y].nunique(dropna=False)
                palette = get_dynamic_palette(num_cat)
                sns.boxplot(x=col_y, y=col_x, data=df, palette=palette, ax=ax)
                ax.set_xlabel(col_y, fontsize=6)
                ax.set_ylabel(col_x, fontsize=6)

            ax.set_title(f"Box Plot: {col_x} vs {col_y}", fontsize=8, weight='bold')
            ax.tick_params(axis='x', rotation=45, labelsize=5)
            ax.tick_params(axis='y', labelsize=5)

        # Atur legend biar kecil
        leg = ax.get_legend()
        if leg:
            leg.set_title(leg.get_title().get_text(), prop={'size': 5})
            for text in leg.get_texts():
                text.set_fontsize(5)

        sns.despine()
        st.pyplot(fig)

    with tab5:
        st.subheader("📈 Time Series By Month")

        # Pastikan kolom bulan dan nilai numerik dipilih user
        month_col = "month"
        value_col = st.selectbox("Select Column Numeric", df.select_dtypes(include='number').columns)

        # Urutkan bulan jika formatnya string (jan, feb, ...)
        month_order = ["jan", "feb", "mar", "apr", "may", "jun", 
                    "jul", "aug", "sep", "oct", "nov", "dec"]
        if df[month_col].dtype == 'object':
            df[month_col] = df[month_col].str.lower()
            df[month_col] = pd.Categorical(df[month_col], categories=month_order, ordered=True)

        # Agregasi per bulan
        df_month = df.groupby(month_col, as_index=False)[value_col].sum().copy()

        # Plot
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.lineplot(x=month_col, y=value_col, data=df_month, marker="o", ax=ax, color="royalblue")
        ax.set_title(f"Trend {value_col} By Month", fontsize=10, weight='bold')
        ax.set_xlabel("Month", fontsize=8)
        ax.set_ylabel(value_col, fontsize=8)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)

        for x, y in zip(df_month[month_col], df_month[value_col]):
            ax.text(x, y, f"{y:,.0f}", fontsize=6, ha='center', va='bottom')

        sns.despine()
        plt.tight_layout()
        st.pyplot(fig)



# ====== PREDICTION ======
elif menu == "Prediction":
    st.title("🤖 Customer Prediction")
    model = model.copy()  # pastikan model tidak berubah

    st.markdown("Enter customer data (manually)")

    # input fields (ambil pilihan dari df_train.unique untuk konsistensi)
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox("Job", df_train['job'].unique())
    marital = st.selectbox("Marital Status", df_train['marital'].unique())
    education = st.selectbox("Education", df_train['education'].unique())
    default = st.selectbox("Default Credit", df_train['default'].unique())
    balance = st.number_input("Balance", min_value=-100000, max_value=1000000, value=300000)
    housing = st.selectbox("Housing Loan", df_train['housing'].unique())
    loan = st.selectbox("Personal Loan", df_train['loan'].unique())
    contact = st.selectbox("Contact Type", df_train['contact'].unique())
    day = st.number_input("Last Contact Day", min_value=1, max_value=31, value=15)
    month = st.selectbox("Last Contact Month", df_train['month'].unique())
    duration = st.number_input("Contact Duration (sec)", min_value=0, max_value=5000, value=100)
    campaign = st.number_input("Number of Contacts", min_value=1, max_value=100, value=10)
    pdays = st.number_input("Days Since Last Contact", min_value=-1, max_value=1000, value=15)
    previous = st.number_input("Previous Contacts", min_value=0, max_value=100, value=9)
    poutcome = st.selectbox("Previous Outcome", df_train['poutcome'].unique())

    input_df = pd.DataFrame([{
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'day': day,
        'month': month,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome
    }])

    # transform kategorikal dengan encoder yang sudah di-fit dari train.csv
    for col in cat_cols:
        input_df[col] = encoders[col].transform(input_df[col])

    # Pastikan urutan kolom sama seperti training (jika model expect feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        model_features = list(model.feature_names_in_)
        # add missing cols default 0
        for c in model_features:
            if c not in input_df.columns:
                input_df[c] = 0
        input_df = input_df[model_features]

    if st.button("Predict"):
        pred = model.predict(input_df)[0]
        # kalau model punya predict_proba
        prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
        if prob is not None:
            st.success(f"Prediction: {pred} (Probability: {prob:.2f})")
        else:
            st.success(f"Prediction: {pred}")
