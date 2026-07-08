"""
Sales Forecasting & Demand Intelligence Dashboard
Run: streamlit run app.py
Requires: train.csv in the same folder (Superstore Sales dataset).
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

FEATURE_COLS = ['lag1', 'lag2', 'lag3', 'rolling_mean_3', 'month', 'quarter', 'season']


# ---------------------------------------------------------------------------
# Data loading & caching
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)
    df['Order Year'] = df['Order Date'].dt.year
    df['Order Month'] = df['Order Date'].dt.month
    return df


def make_monthly_features(monthly_series):
    feat = pd.DataFrame({'Sales': monthly_series})
    feat['lag1'] = feat['Sales'].shift(1)
    feat['lag2'] = feat['Sales'].shift(2)
    feat['lag3'] = feat['Sales'].shift(3)
    feat['rolling_mean_3'] = feat['Sales'].shift(1).rolling(3).mean()
    feat['month'] = feat.index.month
    feat['quarter'] = feat.index.quarter
    feat['season'] = feat.index.month % 12 // 3
    return feat.dropna()


@st.cache_resource
def train_forecast_model(monthly_series, holdout=3):
    """Trains an XGBoost model on all-but-holdout months and returns model, metrics, and full-horizon forecaster."""
    feat = make_monthly_features(monthly_series)
    X, y = feat[FEATURE_COLS], feat['Sales']

    if len(X) <= holdout + 3:
        holdout = max(1, len(X) - 3)

    X_train, y_train = X.iloc[:-holdout], y.iloc[:-holdout]
    X_test, y_test = X.iloc[-holdout:], y.iloc[-holdout:]

    model = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    test_preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, test_preds)
    rmse = np.sqrt(mean_squared_error(y_test, test_preds))

    # refit on full data for genuine future forecasting
    full_model = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    full_model.fit(X, y)

    return full_model, mae, rmse


def forecast_future(monthly_series, model, steps=3):
    history = list(monthly_series.values)
    last_date = monthly_series.index[-1]
    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=steps, freq='MS')
    preds = []
    for d in future_dates:
        last_vals = history[-3:]
        row = {'lag1': last_vals[-1], 'lag2': last_vals[-2], 'lag3': last_vals[-3],
               'rolling_mean_3': np.mean(last_vals), 'month': d.month,
               'quarter': d.quarter, 'season': d.month % 12 // 3}
        p = model.predict(pd.DataFrame([row])[FEATURE_COLS])[0]
        preds.append(p)
        history.append(p)
    return pd.Series(preds, index=future_dates)


@st.cache_data
def compute_anomalies(df):
    weekly = df.set_index('Order Date').resample('W')['Sales'].sum().asfreq('W').fillna(0)
    weekly_df = weekly.to_frame()

    iso = IsolationForest(contamination=0.07, random_state=42)
    weekly_df['iso_anomaly'] = iso.fit_predict(weekly_df[['Sales']]) == -1

    window = 8
    weekly_df['rolling_mean'] = weekly_df['Sales'].rolling(window, center=True, min_periods=1).mean()
    weekly_df['rolling_std'] = weekly_df['Sales'].rolling(window, center=True, min_periods=1).std()
    weekly_df['z_score'] = (weekly_df['Sales'] - weekly_df['rolling_mean']) / weekly_df['rolling_std']
    weekly_df['z_anomaly'] = weekly_df['z_score'].abs() > 2
    return weekly_df


@st.cache_data
def compute_clusters(df):
    total_sales = df.groupby('Sub-Category')['Sales'].sum().rename('total_sales')
    avg_order_value = df.groupby('Sub-Category')['Sales'].mean().rename('avg_order_value')
    monthly_by_sub = df.set_index('Order Date').groupby('Sub-Category').resample('MS')['Sales'].sum()
    volatility = monthly_by_sub.groupby('Sub-Category').std().rename('volatility')

    yearly_by_sub = df.groupby(['Sub-Category', 'Order Year'])['Sales'].sum().reset_index()

    def yoy_growth(g):
        g = g.sort_values('Order Year')
        if len(g) < 2 or g['Sales'].iloc[0] == 0:
            return np.nan
        return (g['Sales'].iloc[-1] - g['Sales'].iloc[0]) / g['Sales'].iloc[0] * 100

    growth = yearly_by_sub.groupby('Sub-Category').apply(yoy_growth).rename('growth_rate_pct')

    features = pd.concat([total_sales, growth, volatility, avg_order_value], axis=1).dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    features['cluster'] = km.fit_predict(X_scaled)

    cluster_stats = features.groupby('cluster')[['total_sales', 'growth_rate_pct', 'volatility']].mean()
    vol_med = cluster_stats['volatility'].median()
    sales_med = cluster_stats['total_sales'].median()
    growth_med = cluster_stats['growth_rate_pct'].median()

    def label_cluster(row):
        if row['total_sales'] >= sales_med and row['volatility'] < vol_med:
            return 'High Volume, Stable Demand'
        elif row['total_sales'] < sales_med and row['volatility'] >= vol_med:
            return 'Low Volume, High Volatility'
        elif row['growth_rate_pct'] >= growth_med:
            return 'Growing Demand'
        else:
            return 'Declining Demand'

    labels = {c: label_cluster(cluster_stats.loc[c]) for c in cluster_stats.index}
    features['cluster_label'] = features['cluster'].map(labels)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    features['pc1'], features['pc2'] = pcs[:, 0], pcs[:, 1]

    return features.reset_index().rename(columns={'index': 'Sub-Category'})


df = load_data()

st.sidebar.title("📦 Sales Intelligence")
page = st.sidebar.radio("Navigate to:", [
    "1️⃣ Sales Overview", "2️⃣ Forecast Explorer", "3️⃣ Anomaly Report", "4️⃣ Product Demand Segments"
])

# ---------------------------------------------------------------------------
# PAGE 1 — Sales Overview
# ---------------------------------------------------------------------------
if page.startswith("1"):
    st.title("📊 Sales Overview Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        region_filter = st.multiselect("Filter by Region", sorted(df['Region'].unique()), default=list(df['Region'].unique()))
    with col2:
        category_filter = st.multiselect("Filter by Category", sorted(df['Category'].unique()), default=list(df['Category'].unique()))

    filtered = df[df['Region'].isin(region_filter) & df['Category'].isin(category_filter)]

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Sales", f"${filtered['Sales'].sum():,.0f}")
    k2.metric("Total Orders", f"{filtered['Order ID'].nunique():,}")
    k3.metric("Avg Order Value", f"${filtered['Sales'].mean():,.2f}")

    yearly = filtered.groupby('Order Year')['Sales'].sum().reset_index()
    fig1 = px.bar(yearly, x='Order Year', y='Sales', title='Total Sales by Year', text_auto='.2s')
    st.plotly_chart(fig1, use_container_width=True)

    monthly_trend = filtered.set_index('Order Date').resample('MS')['Sales'].sum().reset_index()
    fig2 = px.line(monthly_trend, x='Order Date', y='Sales', title='Monthly Sales Trend', markers=True)
    st.plotly_chart(fig2, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        by_region = filtered.groupby('Region')['Sales'].sum().reset_index()
        fig3 = px.pie(by_region, names='Region', values='Sales', title='Sales by Region')
        st.plotly_chart(fig3, use_container_width=True)
    with c2:
        by_cat = filtered.groupby('Category')['Sales'].sum().reset_index()
        fig4 = px.bar(by_cat, x='Category', y='Sales', title='Sales by Category', color='Category')
        st.plotly_chart(fig4, use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE 2 — Forecast Explorer
# ---------------------------------------------------------------------------
elif page.startswith("2"):
    st.title("🔮 Forecast Explorer")

    dim = st.selectbox("Select dimension", ["Category", "Region"])
    options = sorted(df[dim].unique())
    choice = st.selectbox(f"Select {dim}", options)
    horizon = st.slider("Forecast horizon (months ahead)", 1, 3, 3)

    subset = df[df[dim] == choice]
    monthly_series = subset.set_index('Order Date').resample('MS')['Sales'].sum().asfreq('MS').fillna(0)

    with st.spinner("Training XGBoost model..."):
        model, mae, rmse = train_forecast_model(monthly_series)
        forecast = forecast_future(monthly_series, model, steps=horizon)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_series.index, y=monthly_series.values, name='Actual', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name='Forecast', mode='lines+markers', line=dict(dash='dash')))
    fig.update_layout(title=f'{horizon}-Month Forecast for {choice} ({dim})', xaxis_title='Date', yaxis_title='Sales ($)')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast values")
    st.dataframe(forecast.rename('Forecasted Sales ($)').reset_index().rename(columns={'index': 'Month'}))

    m1, m2 = st.columns(2)
    m1.metric("Model MAE (holdout)", f"${mae:,.0f}")
    m2.metric("Model RMSE (holdout)", f"${rmse:,.0f}")
    st.caption("Model: XGBoost (recommended in Task 3 based on lowest MAPE on the overall company-level holdout test).")

# ---------------------------------------------------------------------------
# PAGE 3 — Anomaly Report
# ---------------------------------------------------------------------------
elif page.startswith("3"):
    st.title("🚨 Anomaly Report")

    weekly_df = compute_anomalies(df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly_df.index, y=weekly_df['Sales'], name='Weekly Sales', mode='lines'))

    iso_only = weekly_df[weekly_df['iso_anomaly'] & ~weekly_df['z_anomaly']]
    z_only = weekly_df[weekly_df['z_anomaly'] & ~weekly_df['iso_anomaly']]
    both = weekly_df[weekly_df['iso_anomaly'] & weekly_df['z_anomaly']]

    fig.add_trace(go.Scatter(x=iso_only.index, y=iso_only['Sales'], mode='markers', name='Isolation Forest only',
                              marker=dict(color='orange', size=10)))
    fig.add_trace(go.Scatter(x=z_only.index, y=z_only['Sales'], mode='markers', name='Z-score only',
                              marker=dict(color='green', size=10, symbol='triangle-up')))
    fig.add_trace(go.Scatter(x=both.index, y=both['Sales'], mode='markers', name='Both methods agree',
                              marker=dict(color='red', size=14, symbol='star')))
    fig.update_layout(title='Weekly Sales Anomalies', xaxis_title='Week', yaxis_title='Sales ($)')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected anomaly weeks")
    anomalies = weekly_df[weekly_df['iso_anomaly'] | weekly_df['z_anomaly']][['Sales', 'iso_anomaly', 'z_anomaly', 'z_score']]
    anomalies = anomalies.sort_values('Sales', ascending=False).reset_index().rename(columns={'Order Date': 'Week'})
    st.dataframe(anomalies, use_container_width=True)

    iso_count = weekly_df['iso_anomaly'].sum()
    z_count = weekly_df['z_anomaly'].sum()
    both_count = (weekly_df['iso_anomaly'] & weekly_df['z_anomaly']).sum()
    st.caption(f"Isolation Forest flagged {iso_count} weeks · Z-score flagged {z_count} weeks · both agree on {both_count} weeks.")

# ---------------------------------------------------------------------------
# PAGE 4 — Product Demand Segments
# ---------------------------------------------------------------------------
elif page.startswith("4"):
    st.title("🧩 Product Demand Segments")

    clusters = compute_clusters(df)

    fig = px.scatter(clusters, x='pc1', y='pc2', color='cluster_label', text='Sub-Category',
                      size='total_sales', hover_data=['total_sales', 'growth_rate_pct', 'volatility'],
                      title='Product Sub-Category Demand Clusters (PCA projection)')
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sub-category to cluster mapping")
    display_cols = ['Sub-Category', 'cluster_label', 'total_sales', 'growth_rate_pct', 'volatility', 'avg_order_value']
    st.dataframe(clusters[display_cols].sort_values('cluster_label'), use_container_width=True)

    st.subheader("Recommended stocking strategy")
    strategy = {
        'High Volume, Stable Demand': 'Maintain steady safety stock, standard reorder points — reliable, predictable movers.',
        'Low Volume, High Volatility': 'Keep lean base stock; plan for occasional large one-off orders via short-lead-time suppliers.',
        'Growing Demand': 'Gradually increase reorder quantities; monitor monthly growth trend closely.',
        'Declining Demand': 'Reduce future purchase orders; consider clearance to avoid overstock capital lock-up.',
    }
    for label, advice in strategy.items():
        st.markdown(f"**{label}:** {advice}")
