import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# SET PAGE CONFIG 
st.set_page_config(
    page_title="Smart CSV Explorer | Data Analysis Tool", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .info-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
    .guide-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📊 Smart CSV Explorer</h1>', unsafe_allow_html=True)
st.markdown("### Your Intelligent Data Analysis & Cleaning Companion")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None

# SIDEBAR CONTROLS
st.sidebar.header("🎛️ Explorer Controls")

# User Guide
with st.sidebar.expander("📖 How to Use", expanded=False):
    st.markdown("""
    **Step 1:** Upload your CSV file
    
    **Step 2:** Review data summary and insights
    
    **Step 3:** Clean data (fix missing values, remove duplicates)
    
    **Step 4:** Explore with visualizations
    
    **Step 5:** Filter and export your cleaned data
    """)

uploaded_file = st.sidebar.file_uploader("📁 Upload CSV File", type=["csv"], help="Upload any CSV file for analysis")

# Helper Functions
def detect_header(df_preview):
    """Smart header detection by checking data types"""
    first_row = df_preview.iloc[0].values
    # Convert to Series to use .notna() method
    numeric_values = pd.Series(pd.to_numeric(first_row, errors='coerce'))
    numeric_count = numeric_values.notna().sum()
    ratio_numeric = numeric_count / len(first_row) if len(first_row) > 0 else 0
    return ratio_numeric < 0.5  # If less than 50% numeric, likely headers

def generate_column_names(num_cols):
    """Generate smart column names"""
    names = []
    for i in range(num_cols):
        names.append(f"Column_{i+1}")
    return names

def analyze_data_quality(df):
    """Comprehensive data quality analysis"""
    quality_report = {
        'total_rows': len(df),
        'total_cols': len(df.columns),
        'missing_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_cols': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_cols': len(df.select_dtypes(include=['object']).columns),
        'date_cols': len(df.select_dtypes(include=['datetime64']).columns)
    }
    return quality_report

def get_data_insights(df):
    """Generate smart insights about the data"""
    insights = []
    
    # Missing values insights
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        insights.append(f"⚠️ **{len(missing_cols)} columns** have missing values")
        for col in missing_cols[:3]:  # Show top 3
            pct = (df[col].isnull().sum() / len(df)) * 100
            insights.append(f"   - `{col}`: {df[col].isnull().sum()} missing ({pct:.1f}%)")
    
    # Duplicate insights
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        pct = (duplicates / len(df)) * 100
        insights.append(f"🔄 Found **{duplicates} duplicate rows** ({pct:.1f}% of data)")
    
    # Outlier insights
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_info = []
    for col in numeric_cols[:3]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
        if outliers > 0:
            outlier_info.append(f"`{col}` has {outliers} outliers")
    if outlier_info:
        insights.append(f"📊 Potential outliers detected in: {', '.join(outlier_info)}")
    
    # Data distribution insights
    for col in numeric_cols[:2]:
        if df[col].skew() > 1:
            insights.append(f"📈 `{col}` is right-skewed")
        elif df[col].skew() < -1:
            insights.append(f"📉 `{col}` is left-skewed")
    
    return insights

# HANDLING UPLOADED FILE
if uploaded_file is not None:
    try:
        # Reset file pointer to beginning
        uploaded_file.seek(0)
        
        # First, try to read and detect if there are headers
        try:
            df_preview = pd.read_csv(uploaded_file, nrows=5)
            uploaded_file.seek(0)  # Reset to beginning after preview read
        except pd.errors.EmptyDataError:
            st.sidebar.error("❌ The file appears to be empty. Please upload a valid CSV file.")
            st.stop()
        except Exception as e:
            st.sidebar.error(f"❌ Error reading file: {e}")
            st.stop()
        
        # Check if we got any data
        if df_preview.empty or len(df_preview.columns) == 0:
            st.sidebar.error("❌ The file appears to have no data or columns. Please check your CSV file.")
            st.stop()
        
        # Smart header detection
        has_header = detect_header(df_preview)
        uploaded_file.seek(0)  # Reset to beginning again
        
        if not has_header:
            st.sidebar.warning("⚠️ No headers detected. Would you like to use the first row as headers?")
            use_first_row = st.sidebar.radio("Header Options", ["Generate column names", "Use first row as headers", "No headers"])
            
            uploaded_file.seek(0)  # Reset file pointer
            if use_first_row == "Generate column names":
                df = pd.read_csv(uploaded_file, header=None, names=generate_column_names(len(df_preview.columns)))
            elif use_first_row == "Use first row as headers":
                df = pd.read_csv(uploaded_file, header=0)
            else:
                df = pd.read_csv(uploaded_file, header=None)
        else:
            uploaded_file.seek(0)  # Reset file pointer
            df = pd.read_csv(uploaded_file)
        
        # Check if we got valid data
        if df.empty:
            st.sidebar.error("❌ The file uploaded but contains no data rows.")
            st.stop()
        
        st.session_state.original_df = df.copy()
        st.session_state.df = df.copy()
        st.session_state.file_name = uploaded_file.name
        
        st.sidebar.success(f"✅ File uploaded successfully! ({len(df)} rows, {len(df.columns)} columns)")
        
    except pd.errors.EmptyDataError:
        st.sidebar.error("❌ The file appears to be empty. Please upload a valid CSV file.")
        st.stop()
    except pd.errors.ParserError as e:
        st.sidebar.error(f"❌ Error parsing CSV file: {e}. Please check your file format.")
        st.stop()
    except Exception as e:
        st.sidebar.error(f"❌ Error uploading file: {e}")
        st.stop()

# MAIN CONTENT AREA
if st.session_state.df is not None:
    df = st.session_state.df
    
    # Dashboard Header with Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    quality = analyze_data_quality(df)
    
    with col1:
        st.metric("📊 Total Rows", f"{quality['total_rows']:,}")
    with col2:
        st.metric("📁 Total Columns", quality['total_cols'])
    with col3:
        st.metric("⚠️ Missing Data", f"{quality['missing_pct']:.1f}%")
    with col4:
        st.metric("🔄 Duplicates", f"{quality['duplicate_rows']:,}")
    
    # Data Quality Insights
    st.markdown("---")
    st.subheader("🔍 Data Quality Insights")
    
    insights = get_data_insights(df)
    if insights:
        st.markdown('<div class="warning-card">', unsafe_allow_html=True)
        for insight in insights:
            st.markdown(insight)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-card">✅ Your data looks clean! No major issues detected.</div>', unsafe_allow_html=True)
    
    # Main Tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Data Preview", "🧹 Data Cleaning", "📊 Visualizations", "🎯 Data Filtering", "💾 Export"])
    
    # TAB 1: Data Preview
    with tab1:
        st.markdown("### Data Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Columns:** {', '.join(df.columns.tolist()[:5])}{'...' if len(df.columns) > 5 else ''}")
        with col2:
            st.markdown(f"**Data Types:** {', '.join([f'{k}: {str(v)}' for k,v in df.dtypes.head(3).items()])}")
        with col3:
            st.markdown(f"**Unique Values:** {', '.join([f'{k}: {v}' for k,v in df.nunique().head(3).items()])}")
        
        st.markdown("#### Preview Data")
        num_preview = st.slider("Number of rows to preview", 5, 50, 10, key="preview_slider")
        st.dataframe(df.head(num_preview), use_container_width=True, height=400)
        
        st.markdown("#### Data Information Summary")
        info_data = {
            'Column': df.columns.tolist(),
            'Data Type': [str(dtype) for dtype in df.dtypes],
            'Non-Null Count': [df[col].notna().sum() for col in df.columns],
            'Null Count': [df[col].isna().sum() for col in df.columns],
            'Unique Values': [df[col].nunique() for col in df.columns]
        }
        info_df = pd.DataFrame(info_data)
        st.dataframe(info_df, use_container_width=True)
        
        st.markdown("#### Statistical Summary")
        st.dataframe(df.describe(include='all').T, use_container_width=True)
    
    # TAB 2: Data Cleaning
    with tab2:
        st.markdown("### 🧹 Data Cleaning Tools")
        
        cleaning_options = st.multiselect(
            "Select cleaning operations to perform:",
            ["Fill Missing Values", "Remove Duplicates", "Remove Outliers", "Convert Data Types", "Remove Columns"],
            default=[]
        )
        
        if "Fill Missing Values" in cleaning_options:
            st.markdown("#### Fill Missing Values")
            
            missing_cols = df.columns[df.isnull().any()].tolist()
            if missing_cols:
                selected_col = st.selectbox("Select column with missing values", missing_cols)
                
                col1, col2 = st.columns(2)
                with col1:
                    strategy = st.radio("Imputation Strategy", ["Mean", "Median", "Mode", "Zero", "Custom Value"])
                
                if strategy == "Mean":
                    fill_value = df[selected_col].mean()
                    st.info(f"Will fill missing values with mean: {fill_value:.2f}")
                elif strategy == "Median":
                    fill_value = df[selected_col].median()
                    st.info(f"Will fill missing values with median: {fill_value:.2f}")
                elif strategy == "Mode":
                    fill_value = df[selected_col].mode()[0] if len(df[selected_col].mode()) > 0 else None
                    st.info(f"Will fill missing values with mode: {fill_value}")
                elif strategy == "Zero":
                    fill_value = 0
                else:
                    fill_value = st.number_input("Enter custom value")
                
                if st.button(f"Apply to {selected_col}"):
                    df[selected_col].fillna(fill_value, inplace=True)
                    st.session_state.df = df
                    st.success(f"✅ Filled missing values in {selected_col}")
                    st.rerun()
            else:
                st.success("✅ No missing values to fill!")
        
        if "Remove Duplicates" in cleaning_options:
            st.markdown("#### Remove Duplicate Rows")
            
            duplicates = df.duplicated().sum()
            st.info(f"Found **{duplicates} duplicate rows** in your dataset")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Remove All Duplicates"):
                    new_df = df.drop_duplicates()
                    st.session_state.df = new_df
                    st.success(f"✅ Removed {duplicates} duplicate rows. {len(new_df)} rows remaining.")
                    st.rerun()
            
            with col2:
                subset = st.multiselect("Or remove duplicates based on specific columns", df.columns.tolist())
                if subset and st.button("Remove Duplicates (Subset)"):
                    new_df = df.drop_duplicates(subset=subset)
                    removed = len(df) - len(new_df)
                    st.session_state.df = new_df
                    st.success(f"✅ Removed {removed} duplicate rows based on {subset}")
                    st.rerun()
        
        if "Remove Outliers" in cleaning_options:
            st.markdown("#### Remove Outliers")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                outlier_col = st.selectbox("Select column", numeric_cols)
                
                method = st.radio("Detection Method", ["IQR (Interquartile Range)", "Z-Score"])
                
                if method == "IQR":
                    q1, q3 = df[outlier_col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = ((df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)).sum()
                    st.info(f"Found {outliers} outliers (outside [{lower_bound:.2f}, {upper_bound:.2f}])")
                    
                    if st.button("Remove Outliers"):
                        df = df[(df[outlier_col] >= lower_bound) & (df[outlier_col] <= upper_bound)]
                        st.session_state.df = df
                        st.success(f"✅ Removed {outliers} outliers")
                        st.rerun()
                else:  # Z-Score
                    threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.5)
                    z_scores = np.abs((df[outlier_col] - df[outlier_col].mean()) / df[outlier_col].std())
                    outliers = (z_scores > threshold).sum()
                    st.info(f"Found {outliers} outliers (|Z| > {threshold})")
                    
                    if st.button("Remove Outliers"):
                        df = df[z_scores <= threshold]
                        st.session_state.df = df
                        st.success(f"✅ Removed {outliers} outliers")
                        st.rerun()
            else:
                st.warning("No numeric columns found for outlier detection")
        
        if "Convert Data Types" in cleaning_options:
            st.markdown("#### Convert Data Types")
            col_name = st.selectbox("Select column", df.columns.tolist())
            new_type = st.selectbox("Convert to", ["int", "float", "str", "datetime"])
            
            if st.button("Convert"):
                try:
                    if new_type == "datetime":
                        df[col_name] = pd.to_datetime(df[col_name])
                    else:
                        df[col_name] = df[col_name].astype(new_type)
                    st.session_state.df = df
                    st.success(f"✅ Converted {col_name} to {new_type}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        if "Remove Columns" in cleaning_options:
            st.markdown("#### Remove Columns")
            cols_to_remove = st.multiselect("Select columns to remove", df.columns.tolist())
            
            if cols_to_remove and st.button("🗑️ Remove Selected Columns"):
                df = df.drop(columns=cols_to_remove)
                st.session_state.df = df
                st.success(f"✅ Removed {len(cols_to_remove)} columns")
                st.rerun()
    
    # TAB 3: Visualizations
    with tab3:
        st.markdown("### 📊 Interactive Visualizations")
        
        viz_type = st.selectbox(
            "Choose visualization type",
            ["📊 Histogram", "📈 Line Chart", "🎨 Scatter Plot", "📊 Bar Chart", "🥧 Pie Chart", "🎪 Box Plot", "💎 Correlation Heatmap"]
        )
        
        if viz_type == "📊 Histogram":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                col = st.selectbox("Select column", numeric_cols)
                bins = st.slider("Number of bins", 10, 100, 30)
                fig = px.histogram(df, x=col, nbins=bins, title=f"Distribution of {col}")
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns available for histogram")
        
        elif viz_type == "📈 Line Chart":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                cols = st.multiselect("Select columns", numeric_cols)
                if cols:
                    fig = px.line(df, y=cols, title="Line Chart")
                    fig.update_layout(template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "🎨 Scatter Plot":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("X-axis", numeric_cols)
                y_col = st.selectbox("Y-axis", [col for col in numeric_cols if col != x_col])
                color_by = st.selectbox("Color by (optional)", ["None"] + df.columns.tolist())
                
                if color_by != "None":
                    fig = px.scatter(df, x=x_col, y=y_col, color=color_by, title=f"{y_col} vs {x_col}")
                else:
                    fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns for scatter plot")
        
        elif viz_type == "📊 Bar Chart":
            cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if cat_cols and numeric_cols:
                cat_col = st.selectbox("Category", cat_cols)
                val_col = st.selectbox("Value", numeric_cols)
                
                agg_type = st.radio("Aggregation", ["Sum", "Mean", "Count"])
                top_n = st.slider("Show top N", 5, 50, 10)
                
                if agg_type == "Sum":
                    grouped = df.groupby(cat_col)[val_col].sum().nlargest(top_n).reset_index()
                elif agg_type == "Mean":
                    grouped = df.groupby(cat_col)[val_col].mean().nlargest(top_n).reset_index()
                else:
                    grouped = df.groupby(cat_col).size().nlargest(top_n).reset_index()
                    val_col = grouped.columns[1]
                
                fig = px.bar(grouped, x=cat_col, y=val_col, title=f"{agg_type} of {val_col} by {cat_col}")
                fig.update_layout(template="plotly_white", xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need categorical and numeric columns for bar chart")
        
        elif viz_type == "🥧 Pie Chart":
            cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            if cat_cols:
                col = st.selectbox("Select column", cat_cols)
                top_n = st.slider("Show top N categories", 3, 20, 10)
                
                value_counts = df[col].value_counts().head(top_n)
                fig = px.pie(df, names=value_counts.index, values=value_counts.values, title=f"Distribution of {col}")
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need categorical columns for pie chart")
        
        elif viz_type == "🎪 Box Plot":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                col = st.selectbox("Select column", numeric_cols)
                by_col = st.selectbox("Group by (optional)", ["None"] + df.select_dtypes(exclude=[np.number]).columns.tolist())
                
                if by_col != "None":
                    fig = px.box(df, x=by_col, y=col, title=f"Box Plot of {col} by {by_col}")
                else:
                    fig = px.box(df, y=col, title=f"Box Plot of {col}")
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns available")
        
        elif viz_type == "💎 Correlation Heatmap":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                fig = px.imshow(df[numeric_cols].corr(), title="Correlation Heatmap", 
                              labels=dict(x="Variables", y="Variables", color="Correlation"),
                              aspect="auto", color_continuous_scale="RdBu")
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns for correlation heatmap")
    
    # TAB 4: Data Filtering
    with tab4:
        st.markdown("### 🎯 Filter & Search Data")
        st.markdown("Use the controls below to filter your dataset")
        
        col1, col2 = st.columns(2)
        
        filters = {}
        with col1:
            selected_col = st.selectbox("Filter by column", df.columns)
        
        with col2:
            if pd.api.types.is_numeric_dtype(df[selected_col]):
                min_val = float(df[selected_col].min())
                max_val = float(df[selected_col].max())
                value_range = st.slider("Select range", min_val, max_val, (min_val, max_val))
                filtered_df = df[(df[selected_col] >= value_range[0]) & (df[selected_col] <= value_range[1])]
            else:
                options = df[selected_col].dropna().unique().tolist()
                selected_options = st.multiselect("Select values", options, default=options)
                filtered_df = df[df[selected_col].isin(selected_options)]
        
        st.markdown(f"**Filtered dataset:** {len(filtered_df)} rows (from {len(df)} original rows)")
        
        # Additional filters
        if st.checkbox("Apply additional filters"):
            for col in df.columns:
                if col != selected_col:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        min_val = float(df[col].min())
                        max_val = float(df[col].max())
                        value_range = st.slider(f"Filter {col}", min_val, max_val, (min_val, max_val), key=f"filter_{col}")
                        filtered_df = filtered_df[(filtered_df[col] >= value_range[0]) & (filtered_df[col] <= value_range[1])]
        
        st.markdown("#### Filtered Data Preview")
        st.dataframe(filtered_df, use_container_width=True, height=400)
        
        st.markdown("#### Download Filtered Data")
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered CSV",
            data=csv,
            file_name=f"filtered_{st.session_state.file_name}",
            mime="text/csv"
        )
    
    # TAB 5: Export
    with tab5:
        st.markdown("### 💾 Export Your Cleaned Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### **Original vs Current**")
            st.metric("Original Rows", len(st.session_state.original_df))
            st.metric("Current Rows", len(df))
            changes = len(st.session_state.original_df) - len(df)
            if changes != 0:
                st.metric("Rows Changed", f"{changes:+d}")
        
        with col2:
            st.markdown("#### **Export Options**")
            export_format = st.radio("Format", ["CSV", "Excel", "JSON"])
        
        with col3:
            st.markdown("#### **Quick Actions**")
            if st.button("🔄 Reset to Original"):
                st.session_state.df = st.session_state.original_df.copy()
                st.success("✅ Reset to original data")
                st.rerun()
        
        st.markdown("---")
        
        if export_format == "CSV":
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data,
                file_name=f"cleaned_{st.session_state.file_name}",
                mime="text/csv"
            )
        elif export_format == "Excel":
            output = pd.ExcelWriter(StringIO(), engine='openpyxl')
            df.to_excel(output, index=False)
            excel_data = output.path
            st.download_button(
                label="📥 Download as Excel",
                data=excel_data,
                file_name=f"cleaned_{st.session_state.file_name.replace('.csv', '.xlsx')}",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:  # JSON
            json_data = df.to_json(orient='records').encode('utf-8')
            st.download_button(
                label="📥 Download as JSON",
                data=json_data,
                file_name=f"cleaned_{st.session_state.file_name.replace('.csv', '.json')}",
                mime="application/json"
            )
        
        st.markdown("---")
        st.markdown("#### **Data Statistics**")
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        with stats_col1:
            st.write("**Numeric Columns:**", len(df.select_dtypes(include=[np.number]).columns))
            st.write("**Categorical Columns:**", len(df.select_dtypes(exclude=[np.number]).columns))
        
        with stats_col2:
            st.write("**Total Missing Values:**", df.isnull().sum().sum())
            st.write("**Duplicate Rows:**", df.duplicated().sum())
        
        with stats_col3:
            st.write("**Memory Usage:**", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")

else:
    # Landing page when no file is uploaded
    st.markdown("""
    <div class="guide-box">
        <h2>Welcome to Smart CSV Explorer! 🚀</h2>
        <p>Your intelligent data analysis and cleaning companion.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>✨ Key Features</h3>
            <ul>
                <li>📊 <strong>Smart Data Analysis:</strong> Automatic insights and quality checks</li>
                <li>🧹 <strong>Data Cleaning:</strong> Fill missing values, remove duplicates</li>
                <li>📈 <strong>Rich Visualizations:</strong> Multiple chart types</li>
                <li>🎯 <strong>Data Filtering:</strong> Filter and search your data</li>
                <li>💾 <strong>Export Options:</strong> Download in CSV, Excel, JSON</li>
                <li>🧠 <strong>Smart Headers:</strong> Auto-detects or generates column names</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🚀 Getting Started</h3>
            <ol>
                <li><strong>Upload:</strong> Click on the file uploader in the sidebar</li>
                <li><strong>Explore:</strong> Navigate through the different tabs</li>
                <li><strong>Clean:</strong> Use the cleaning tools to fix your data</li>
                <li><strong>Visualize:</strong> Create charts to understand patterns</li>
                <li><strong>Filter:</strong> Narrow down to the data you need</li>
                <li><strong>Export:</strong> Download your cleaned dataset</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 💡 Tips for Best Results")
    
    tip_col1, tip_col2, tip_col3 = st.columns(3)
    
    with tip_col1:
        st.markdown("""
        **📋 Data Format Tips:**
        - Supports standard CSV files
        - Auto-detects headers
        - Handles missing values gracefully
        """)
    
    with tip_col2:
        st.markdown("""
        **🧹 Cleaning Tips:**
        - Fill missing values with appropriate strategies
        - Remove duplicates before analysis
        - Handle outliers for better insights
        """)
    
    with tip_col3:
        st.markdown("""
        **📊 Visualization Tips:**
        - Choose appropriate chart types
        - Filter data for focused analysis
        - Export for further processing
        """)