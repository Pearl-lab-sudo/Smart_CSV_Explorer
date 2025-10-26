# 📊 Smart CSV Explorer

A powerful, intelligent CSV data exploration and cleaning tool built with Streamlit.

## ✨ Features

### 🧠 Smart Data Handling
- **Auto-Header Detection**: Automatically detects if your CSV has headers
- **Smart Column Naming**: Generates meaningful column names when headers are missing
- **Flexible Loading**: Handles CSVs with or without headers seamlessly

### 📊 Comprehensive Data Analysis
- **Quick Quality Metrics**: View data quality at a glance (rows, columns, missing data, duplicates)
- **Data Insights**: Automatic detection of:
  - Missing values with percentages
  - Duplicate rows
  - Potential outliers
  - Data distribution and skewness
- **Statistical Summary**: Full descriptive statistics for all columns

### 🧹 Advanced Data Cleaning
- **Fill Missing Values**: Choose from multiple strategies:
  - Mean imputation
  - Median imputation
  - Mode (most frequent) imputation
  - Zero fill
  - Custom value
- **Remove Duplicates**: Remove all duplicates or based on specific columns
- **Outlier Detection & Removal**: Two methods:
  - IQR (Interquartile Range) method
  - Z-Score method with adjustable threshold
- **Data Type Conversion**: Convert columns to int, float, str, or datetime
- **Column Removal**: Delete unwanted columns

### 📈 Rich Visualizations
- **Histogram**: View distributions of numeric data
- **Line Charts**: Plot trends over time or categories
- **Scatter Plots**: Explore relationships between variables
- **Bar Charts**: Compare categories with customizable aggregation (sum, mean, count)
- **Pie Charts**: Visualize categorical distributions
- **Box Plots**: Identify outliers and distributions (optional grouping)
- **Correlation Heatmap**: Explore correlations between numeric variables

### 🎯 Advanced Filtering
- **Multi-Column Filtering**: Filter data based on one or more columns
- **Range Filtering**: Numeric filters with sliders
- **Value Selection**: Multi-select for categorical data
- **Real-time Preview**: See filtered results instantly

### 💾 Flexible Export
- **Multiple Formats**: Export as CSV, Excel, or JSON
- **Track Changes**: Compare original vs cleaned data
- **Reset Functionality**: Revert to original data anytime

### 🎨 Modern UI/UX
- **Beautiful Design**: Gradient headers, card-based layout
- **Interactive Guides**: Built-in help and navigation tips
- **Color-Coded Metrics**: Visual indicators for data quality
- **Tabbed Interface**: Organized, easy-to-navigate sections

## 🚀 Getting Started

### Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run CSV_explorer.py
```

3. Open your browser and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

### Quick Start Guide

1. **Upload**: Click "Upload CSV File" in the sidebar and select your CSV file
2. **Explore**: Check the Data Preview tab to understand your data
3. **Analyze**: Review the Data Quality Insights for issues
4. **Clean**: Use the Data Cleaning tab to fix problems:
   - Fill missing values with appropriate strategies
   - Remove duplicate rows
   - Remove outliers if needed
5. **Visualize**: Create charts in the Visualizations tab to discover patterns
6. **Filter**: Use the Data Filtering tab to narrow down your dataset
7. **Export**: Download your cleaned data in your preferred format

## 💡 Tips for Best Results

### Data Format
- Use standard CSV format
- First row as headers (if applicable)
- Consistent data types within columns

### Data Cleaning
- **Missing Values**: Choose imputation strategy based on data type:
  - For numeric: Mean or Median
  - For categorical: Mode (most frequent)
- **Duplicates**: Check if duplicates are meaningful before removal
- **Outliers**: Investigate outliers before removal - they might be important data points

### Visualization
- **Distribution**: Use histograms for understanding data spread
- **Relationships**: Use scatter plots for correlation analysis
- **Comparisons**: Use bar charts for comparing categories
- **Groups**: Use box plots to compare distributions across groups

## 📦 Dependencies

- `streamlit` - Web framework
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `plotly` - Interactive visualizations
- `matplotlib` - Additional plotting capabilities
- `openpyxl` - Excel file support

## 🎯 Use Cases

- **Data Exploration**: Quick EDA (Exploratory Data Analysis)
- **Data Cleaning**: Prepare data for machine learning
- **Data Quality Checks**: Identify issues in datasets
- **Quick Visualizations**: Generate insights on-the-fly
- **Data Export**: Clean and export data for further processing

## 🛠️ Features Breakdown

### Smart Header Detection
The tool intelligently detects headers by analyzing data types in the first row:
- If <50% of values are numeric → treats as headers
- Else → treats as data

### Data Quality Insights
Automatically identifies:
- Missing value patterns
- Duplicate entries
- Statistical outliers
- Data skewness

### Interactive Visualizations
All charts are interactive Plotly visualizations with:
- Hover tooltips
- Zoom and pan capabilities
- Download options

## 🤝 Contributing

Feel free to suggest improvements or report issues!

## 📝 License

This project is open source and available for use and modification.

---

**Made with ❤️ for Data Scientists and Analysts**


