import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os
import joblib
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Supply Chain Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

def create_banner():
    st.markdown("""
    <style>
        .banner-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background: linear-gradient(to right, #1a2980, #26d0ce);
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .banner-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: white;
            margin-bottom: 0.5rem;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        }
        
        .banner-subtitle {
            font-size: 1.2rem;
            color: rgba(255, 255, 255, 0.9);
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .banner-icons {
            display: flex;
            justify-content: center;
            margin-top: 1rem;
        }
        
        .banner-icon {
            font-size: 2rem;
            margin: 0 1rem;
            color: white;
        }
        
        @media (max-width: 768px) {
            .banner-title {
                font-size: 2rem;
            }
            .banner-subtitle {
                font-size: 1rem;
            }
        }
    </style>
    
    <div class="banner-container">
        <div class="banner-title">📦 Supply Chain Analytics</div>
        <div class="banner-subtitle">Interactive insights & predictive analysis for optimized supply chain management</div>
        <div class="banner-icons">
            <div class="banner-icon">🚚</div>
            <div class="banner-icon">📊</div>
            <div class="banner-icon">⏱️</div>
            <div class="banner-icon">📈</div>
            <div class="banner-icon">🔍</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Add this near the top of your app, after setting page config
create_banner()



def create_stylish_sidebar(data):
    st.sidebar.markdown("""
    <style>
        /* Overall sidebar styling - applying gradient */
        section[data-testid="stSidebar"] {
            background: linear-gradient(to bottom, #1a2980, #26d0ce);
            color: white !important;
        }
        
        /* Force all text elements in sidebar to be white */
        section[data-testid="stSidebar"] div, 
        section[data-testid="stSidebar"] span, 
        section[data-testid="stSidebar"] label, 
        section[data-testid="stSidebar"] p {
            color: white !important;
        }
        
        /* Sidebar header */
        .sidebar-header {
            text-align: center;
            font-size: 1.8rem;
            color: white !important;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid white;
            font-weight: bold;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
        }
        
        /* Navigation container */
        div.stRadio > div {
            background-color: transparent;
            padding: 0;
        }
        
        /* Navigation items - enhanced contrast */
        div.stRadio label {
            background-color: rgba(26, 41, 128, 0.5); /* Darker background */
            color: white !important;
            padding: 12px 15px;
            border-radius: 6px;
            margin: 8px 0;
            display: flex;
            align-items: center;
            font-weight: 500; /* Slightly bolder */
            transition: all 0.2s;
            text-shadow: 0px 1px 2px rgba(0,0,0,0.3); /* Text shadow for better readability */
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Navigation items text */
        div.stRadio label span {
            color: white !important;
        }
        
        /* Hover state */
        div.stRadio label:hover {
            background-color: rgba(26, 41, 128, 0.8);
            transform: translateX(5px);
        }
        
        /* Selected state */
        div.stRadio label[data-baseweb="radio"] input:checked + div {
            background-color: white !important;
            border-color: white !important;
        }
        
        div.stRadio label[data-baseweb="radio"] input:checked + div + span {
            color: white !important;
            font-weight: bold !important;
            text-shadow: 0px 1px 2px rgba(0,0,0,0.5);
        }
        
        /* Force radio button colors */
        div.stRadio label[data-baseweb="radio"] div[role="radioitem"] {
            background-color: white !important;
        }
        
        /* Metrics container */
        .metrics-container {
            background-color: rgba(26, 41, 128, 0.5); /* Darker for contrast */
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }
        
        /* Individual metrics */
        .metric-mini {
            text-align: center;
            margin-bottom: 10px;
            padding: 8px 5px;
            background-color: rgba(26, 41, 128, 0.7); /* Darker for better contrast */
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .metric-mini-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: white !important;
            text-shadow: 0px 1px 2px rgba(0,0,0,0.3);
        }
        
        .metric-mini-label {
            font-size: 0.85rem;
            color: white !important;
            margin-top: 4px;
            text-shadow: 0px 1px 1px rgba(0,0,0,0.2);
        }
        
        /* Override any theme-based text coloring throughout the sidebar */
        [data-testid="stSidebar"] [data-testid="stText"],
        [data-testid="stSidebar"] [data-testid="stMarkdown"] p,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stSelectbox span {
            color: white !important;
        }
        
        /* Dropdown menu and options */
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: rgba(255, 255, 255, 0.15) !important;
            color: white !important;
        }
        
        .stSelectbox div[data-baseweb="select"] > div > div {
            color: white !important;
        }
        
        /* Footer */
        .sidebar-footer {
            text-align: center;
            margin-top: 20px;
            padding-top: 10px;
            border-top: 1px solid rgba(255, 255, 255, 0.4);
            font-size: 0.8rem;
            color: white !important;
            text-shadow: 0px 1px 1px rgba(0,0,0,0.2);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # App logo and title
    st.sidebar.image(os.path.join('app', 'funko.png'), use_container_width=True)
    st.sidebar.markdown('<div class="sidebar-header">Supply Chain Analytics</div>', unsafe_allow_html=True)
    
    # Mini metrics dashboard in sidebar
    st.sidebar.markdown('<div class="metrics-container">', unsafe_allow_html=True)
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        otif_pct = data['OTIF'].mean() * 100
        st.markdown(f"""
        <div class="metric-mini">
            <div class="metric-mini-value">{otif_pct:.1f}%</div>
            <div class="metric-mini-label">OTIF Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        on_time_pct = data['ontime'].mean() * 100
        st.markdown(f"""
        <div class="metric-mini">
            <div class="metric-mini-value">{on_time_pct:.1f}%</div>
            <div class="metric-mini-label">On-Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation options with icons
    pages_with_icons = [
        "📋 Project Overview",
        "📊 Dashboard",
        "💼 Power BI Dashboard",
        "🔗 Correlation Analysis",
        "🏭 Supplier Analysis",
        "📦 Product Analysis", 
        "🔮 OTIF Prediction"
    ]
    
    selection = st.sidebar.radio("Navigation", pages_with_icons, label_visibility="collapsed")
    
    # Remove the icon to get the actual page name
    page = selection.split(" ", 1)[1]
    
    # Add footer
    st.sidebar.markdown("""
    <div class="sidebar-footer">
        Supply Chain Analytics Platform v1.0<br>
        Created by Jotis<br>
        &copy; 2025 All Rights Reserved
    </div>
    """, unsafe_allow_html=True)
    
    return page



# Add custom CSS
def load_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: white;
            text-align: center;
            margin-bottom: 1rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid #3498db;
        }
        .sub-header {
            font-size: 1.8rem;
            color: #2c3e50;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .section-header {
            font-size: 1.5rem;
            color: #3498db;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        .highlight {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #3498db;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 0.15rem 0.5rem rgba(0, 0, 0, 0.1);
            text-align: center;
            margin-bottom: 1rem;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #3498db;
        }
        .metric-label {
            font-size: 1rem;
            color: #7f8c8d;
        }
        .prediction-box {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 0.15rem 0.5rem rgba(0, 0, 0, 0.1);
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .prediction-positive {
            color: #27ae60;
            font-weight: bold;
            font-size: 1.5rem;
        }
        .prediction-negative {
            color: #e74c3c;
            font-weight: bold;
            font-size: 1.5rem;
        }
        .footer {
            text-align: center;
            color: #7f8c8d;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #ecf0f1;
        }
    </style>
    """, unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        # Convert date columns to datetime
        date_columns = [col for col in data.columns if 'fecha' in col]
        for col in date_columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')
        return data
    else:
        st.error(f"Data file not found: {file_path}")
        return None

# Visualization functions (keeping existing ones)
def plot_histogram(data, column, title, xlabel):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data[column], kde=True, ax=ax, color='skyblue')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    st.pyplot(fig)

def plot_pie_chart(data, column, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    data[column].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=sns.color_palette('pastel'))
    ax.set_title(title, fontsize=16)
    ax.set_ylabel('')
    st.pyplot(fig)

def plot_line(data, x_column, y_column, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=data, x=x_column, y=y_column, ax=ax, color='coral', linewidth=2.5)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def plot_box(data, x_column, y_column, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, x=x_column, y=y_column, ax=ax, palette='coolwarm')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def plot_bar(data, x_column, y_column, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=data, x=x_column, y=y_column, ax=ax, palette='Blues_d')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    st.pyplot(fig)

def plot_count(data, column, title, xlabel):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=data, x=column, ax=ax, palette='Set2')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    st.pyplot(fig)

def plot_heatmap(data, columns, title):
    fig, ax = plt.subplots(figsize=(12, 10))
    correlation = data[columns].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)

# Function to display the main image
def display_main_image(image_path):
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.error(f"Image not found: {image_path}")

# Function to display the video
def display_video(video_path):
    if os.path.exists(video_path):
        st.video(video_path)
    else:
        st.error(f"Video not found: {video_path}")

# ML model functions
@st.cache_resource
def load_ml_model(model_path='notebook/best_otif_model.pkl'):
    """Load the trained ML model"""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

def predict_otif(model, input_data):
    """Make predictions using the trained model"""
    if model is None:
        st.error("Model not found. Please check if the model file exists.")
        return None, None
    
    try:
        # Ensure the input data has exactly the features the model expects
        expected_features = [
            'cant_prod_odc', 'prec_unt', 'monto_odc', 'cant_recibida', 
            'monto_recibido', 'costo_prod', 'reception_days',
            'amount_difference', 'total_amount', 'delivery_time_diff'
        ]
        
        # Check if all expected features are present
        missing_features = [feat for feat in expected_features if feat not in input_data.columns]
        if missing_features:
            st.error(f"Missing features: {missing_features}")
            return None, None
            
        # Make prediction
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)[:, 1]
        
        return predictions, probabilities
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        # Print out the columns for debugging
        st.warning("Debugging info:")
        st.warning(f"Input data columns: {input_data.columns.tolist()}")
        return None, None

# Function to display key metrics
def display_key_metrics(data):
    #st.markdown('<div class="sub-header" style="color: white;">Key Supply Chain Metrics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        otif_percentage = (data['OTIF'].mean() * 100)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{otif_percentage:.1f}%</div>
                <div class="metric-label">OTIF Rate</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        ontime_percentage = (data['ontime'].mean() * 100)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{ontime_percentage:.1f}%</div>
                <div class="metric-label">On-Time Rate</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col3:
        avg_delivery_days = data['delivery_days'].mean()
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{avg_delivery_days:.1f}</div>
                <div class="metric-label">Avg. Delivery Days</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col4:
        # Calculate cost efficiency
        if 'monto_odc' in data.columns and 'cant_prod_odc' in data.columns:
            # Calculate cost per unit for each order
            plot_data = data.copy()
            plot_data['cost_per_unit'] = plot_data['monto_odc'] / plot_data['cant_prod_odc']
            
            # Remove outliers
            upper_limit = np.percentile(plot_data['cost_per_unit'].dropna(), 95)
            plot_data = plot_data[plot_data['cost_per_unit'] <= upper_limit]
            
            # Calculate cost efficiency metrics
            avg_cost = plot_data['cost_per_unit'].mean()
            min_cost = plot_data['cost_per_unit'].min()
            max_cost = plot_data['cost_per_unit'].max()
            
            # Convert to efficiency score (0-100%)
            # Lower cost = higher efficiency, so we invert the scale
            # If cost is at minimum, efficiency is 100%
            # If cost is at maximum, efficiency is 0%
            cost_range = max_cost - min_cost
            if cost_range > 0:
                efficiency_score = 100 * (1 - ((avg_cost - min_cost) / cost_range))
            else:
                efficiency_score = 100  # If all costs are the same
                
            # Ensure the score is within 0-100 range
            efficiency_score = max(0, min(100, efficiency_score))
        else:
            # Fallback if we don't have the necessary columns
            efficiency_score = 75  # Default value
        
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{efficiency_score:.1f}%</div>
                <div class="metric-label">Cost Efficiency</div>
            </div>
            """, 
            unsafe_allow_html=True
        )

# Create a prediction form
def create_prediction_form(data, model):
    st.markdown('<div class="dashboard-section-title">OTIF Prediction Tool</div>', unsafe_allow_html=True)
    
    # Add CSS for prediction form styling with improved selection elements
    st.markdown("""
    <style>
        /* Form container */
        .stForm {
            background: linear-gradient(135deg, rgba(26, 41, 128, 0.05), rgba(38, 208, 206, 0.05));
            padding: 20px;
            border-radius: 10px;
            border: 1px solid rgba(38, 208, 206, 0.3);
            margin-bottom: 25px;
        }
        
        /* Form section headers */
        .stForm h3 {
            color: white !important;
            background: linear-gradient(to right, rgba(26, 41, 128, 0.8), rgba(38, 208, 206, 0.8));
            padding: 8px 15px;
            border-radius: 8px;
            font-size: 1.2rem;
            margin: 15px 0 15px 0;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
        }
        
        /* Form inputs */
        .stNumberInput, .stSelectbox {
            margin-bottom: 15px;
        }
        
        /* Input labels */
        .stNumberInput label p, .stSelectbox label p, .stRadio label p {
            color: inherit !important;
            font-weight: 500 !important;
            margin-bottom: 5px !important;
        }
        
        /* Help text */
        .stNumberInput div[data-baseweb="help-helper"] {
            color: inherit !important;
            opacity: 0.8;
        }
        
        /* Selection cards container */
        .selection-cards {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 10px 0;
        }
        
        /* Selection card styling */
        .selection-card {
            background: white;
            color: #333;
            padding: 8px 16px;
            border-radius: 6px;
            border: 1px solid rgba(38, 208, 206, 0.5);
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.2s;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 200px;
        }
        
        /* Selection card hover */
        .selection-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            border-color: #1a2980;
        }
        
        /* Active selection card */
        .selection-card.active {
            background: linear-gradient(to right, #1a2980, #26d0ce);
            color: white;
            font-weight: bold;
            border: none;
        }
        
        /* Submit button */
        .stButton button {
            background: linear-gradient(to right, #1a2980, #26d0ce) !important;
            color: white !important;
            font-weight: bold !important;
            padding: 10px 20px !important;
            border-radius: 5px !important;
            border: none !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
            transition: all 0.3s ease !important;
        }
        
        /* Submit button hover */
        .stButton button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15) !important;
        }
        
        /* Search input styling */
        .search-box {
            padding: 10px 15px;
            border-radius: 8px;
            border: 2px solid rgba(38, 208, 206, 0.5);
            width: 100%;
            font-size: 16px;
            margin-bottom: 15px;
            background-color: white;
            color: #333;
        }
        
        /* Prediction results */
        .prediction-box {
            background: linear-gradient(135deg, rgba(26, 41, 128, 0.1), rgba(38, 208, 206, 0.1));
            padding: 20px;
            border-radius: 10px;
            border: 1px solid rgba(38, 208, 206, 0.3);
            margin: 20px 0;
        }
        
        /* Prediction result cards */
        .prediction-positive, .prediction-negative {
            font-size: 1.3rem;
            font-weight: bold;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            text-align: center;
        }
        
        .prediction-positive {
            background-color: rgba(25, 135, 84, 0.2);
            border: 1px solid rgba(25, 135, 84, 0.5);
            color: #198754;
        }
        
        .prediction-negative {
            background-color: rgba(220, 53, 69, 0.2);
            border: 1px solid rgba(220, 53, 69, 0.5);
            color: #dc3545;
        }
        
        /* Result metrics */
        .result-metric {
            background: linear-gradient(135deg, rgba(26, 41, 128, 0.05), rgba(38, 208, 206, 0.05));
            padding: 10px 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border: 1px solid rgba(38, 208, 206, 0.2);
        }
        
        /* Order summary */
        .order-summary {
            background: linear-gradient(135deg, rgba(26, 41, 128, 0.05), rgba(38, 208, 206, 0.05));
            padding: 15px;
            border-radius: 8px;
            border: 1px solid rgba(38, 208, 206, 0.2);
        }
        
        /* Selection indicator */
        .selection-indicator {
            background-color: rgba(255, 255, 255, 0.95);
            color: #333;
            border: 1px solid rgba(38, 208, 206, 0.5);
            border-radius: 6px;
            padding: 10px 15px;
            margin: 10px 0;
            font-weight: 500;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Introduction with styled info box
    st.markdown("""
    <div style="background-color: rgba(26, 41, 128, 0.1); padding: 15px; border-radius: 8px; border: 1px solid rgba(38, 208, 206, 0.3); margin-bottom: 20px;">
        <div style="display: flex; align-items: center;">
            <div style="font-size: 1.5rem; margin-right: 10px; color: #1a2980;">ℹ️</div>
            <div>
                This tool predicts whether a delivery will be On Time In Full (OTIF) based on order parameters. Fill in the form below to generate a prediction.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get unique values for categorical fields
    suppliers = sorted(data['nom_prov'].unique())
    countries = sorted(data['org_pais'].unique())
    categories = sorted(data['Categoria'].unique())
    subcategories = sorted(data['Subcategoria'].unique())
    product_units = sorted(data['und_prod'].unique())
    
    # STEP 1: Category Selection with visual buttons
    st.markdown('<div class="dashboard-section-title" style="font-size: 1.2rem;">Step 1: Select Product Category</div>', unsafe_allow_html=True)
    
    # Initialize session state for selections
    if 'selected_category' not in st.session_state:
        st.session_state.selected_category = categories[0]
    
    # Show the top 6 most common categories as buttons
    top_categories = categories[:6]
    
    # Create 3 columns for buttons layout
    col1, col2, col3 = st.columns(3)
    
    # Distribute top categories across columns
    with col1:
        for idx, cat in enumerate(top_categories[::3]):
            if st.button(f"{cat}", key=f"cat_{idx*3}"):
                st.session_state.selected_category = cat
                
    with col2:
        for idx, cat in enumerate(top_categories[1::3]):
            if st.button(f"{cat}", key=f"cat_{idx*3+1}"):
                st.session_state.selected_category = cat
                
    with col3:
        for idx, cat in enumerate(top_categories[2::3]):
            if st.button(f"{cat}", key=f"cat_{idx*3+2}"):
                st.session_state.selected_category = cat
    
    # Add a "More categories" expander for the remaining categories
    if len(categories) > 6:
        with st.expander("More categories..."):
            # Create a search box for categories
            cat_search = st.text_input("Search for category", key="cat_search", placeholder="Type to search categories...")
            
            # Filter categories based on search
            if cat_search:
                filtered_cats = [c for c in categories[6:] if cat_search.lower() in c.lower()]
            else:
                filtered_cats = categories[6:]
            
            # Display filtered categories in a grid (3 columns)
            cols = st.columns(3)
            for i, cat in enumerate(filtered_cats):
                with cols[i % 3]:
                    if st.button(f"{cat}", key=f"more_cat_{i}"):
                        st.session_state.selected_category = cat
    
    # Display selected category
    selected_category = st.session_state.selected_category
    st.markdown(f"""
    <div class="selection-indicator">
        <strong>Selected Category:</strong> {selected_category}
    </div>
    """, unsafe_allow_html=True)
    
    # STEP 2: Product Selection with search functionality
    st.markdown('<div class="dashboard-section-title" style="font-size: 1.2rem;">Step 2: Select Product</div>', unsafe_allow_html=True)
    
    # Filter products based on selected category
    filtered_products = sorted(data[data['Categoria'] == selected_category]['descrip_prod'].unique())
    
    # Initialize selected product if needed
    if 'selected_product' not in st.session_state or st.session_state.selected_product not in filtered_products:
        st.session_state.selected_product = filtered_products[0] if filtered_products else ""
    
    # Product search for better usability
    product_search = st.text_input("Search product by name", placeholder="Type to search products...", key="product_search")
    
    # Filter products based on search
    if product_search:
        display_products = [p for p in filtered_products if product_search.lower() in p.lower()]
        if not display_products:
            st.warning(f"No products found matching '{product_search}'. Showing top products instead.")
            display_products = filtered_products[:10]
    else:
        # Only show top 10 products if no search
        display_products = filtered_products[:10]
    
    # Display products as a grid of buttons (max 10)
    cols = st.columns(2)  # 2 columns for product buttons
    for i, product in enumerate(display_products):
        with cols[i % 2]:
            button_label = product[:30] + "..." if len(product) > 30 else product
            if st.button(button_label, key=f"prod_{i}"):
                st.session_state.selected_product = product
    
    # If there are more products, show an expander
    if len(filtered_products) > 10 and not product_search:
        with st.expander("Show more products"):
            # Create 2 more columns in the expander
            more_cols = st.columns(2)
            for i, product in enumerate(filtered_products[10:30]):  # Show next 20 products
                with more_cols[i % 2]:
                    button_label = product[:30] + "..." if len(product) > 30 else product
                    if st.button(button_label, key=f"more_prod_{i}"):
                        st.session_state.selected_product = product
            
            # If still more products, add a note
            if len(filtered_products) > 30:
                st.caption(f"{len(filtered_products) - 30} more products available. Use the search box to find specific products.")
    
    # Display selected product
    selected_product = st.session_state.selected_product
    st.markdown(f"""
    <div class="selection-indicator">
        <strong>Selected Product:</strong> {selected_product}
    </div>
    """, unsafe_allow_html=True)
    
    # STEP 3: Prediction Form
    st.markdown('<div class="dashboard-section-title" style="font-size: 1.2rem;">Step 3: Enter Order Details</div>', unsafe_allow_html=True)
    
    # Initialize supplier selection if needed
    if 'selected_supplier' not in st.session_state:
        st.session_state.selected_supplier = suppliers[0]
    
    # Supplier selection with search and top options
    st.markdown("#### Select Supplier")
    supplier_search = st.text_input("Search for supplier", placeholder="Type to search suppliers...", key="supplier_search")
    
    # Filter suppliers based on search
    if supplier_search:
        display_suppliers = [s for s in suppliers if supplier_search.lower() in s.lower()]
        if not display_suppliers:
            st.warning(f"No suppliers found matching '{supplier_search}'. Showing top suppliers instead.")
            display_suppliers = suppliers[:10]
    else:
        # Only show top 8 suppliers if no search
        display_suppliers = suppliers[:8]
    
    # Display suppliers as a grid of buttons
    cols = st.columns(4)  # 4 columns for supplier buttons
    for i, supplier in enumerate(display_suppliers):
        with cols[i % 4]:
            button_label = supplier[:20] + "..." if len(supplier) > 20 else supplier
            if st.button(button_label, key=f"sup_{i}"):
                st.session_state.selected_supplier = supplier
    
    # If there are more suppliers, show an expander
    if len(suppliers) > 8 and not supplier_search:
        with st.expander("More suppliers"):
            # Create 4 more columns in the expander
            more_cols = st.columns(4)
            for i, supplier in enumerate(suppliers[8:24]):  # Show next 16 suppliers
                with more_cols[i % 4]:
                    button_label = supplier[:20] + "..." if len(supplier) > 20 else supplier
                    if st.button(button_label, key=f"more_sup_{i}"):
                        st.session_state.selected_supplier = supplier
    
    # Display selected supplier
    selected_supplier = st.session_state.selected_supplier
    st.markdown(f"""
    <div class="selection-indicator">
        <strong>Selected Supplier:</strong> {selected_supplier}
    </div>
    """, unsafe_allow_html=True)
    
    # Display product units as simple radio buttons (usually short list)
    st.markdown("#### Select Unit")
    selected_unit = st.radio("Unit", product_units, horizontal=len(product_units) <= 6)
    
    # Actual prediction form for numerical inputs
    with st.form("prediction_form"):
        # Basic order information (category 1)
        st.subheader("Order Information")
        col1, col2 = st.columns(2)
        
        with col1:
            # First required model feature
            quantity = st.number_input("Quantity Ordered (cant_prod_odc)", min_value=1, value=100)
            # Second required model feature
            unit_price = st.number_input("Unit Price (prec_unt)", min_value=0.1, value=50.0, step=0.1)
            # Third required model feature (calculated)
            order_amount = st.number_input("Order Amount (monto_odc)", 
                                        value=quantity*unit_price, 
                                        help="Total order amount (quantity × price)")
        
        with col2:
            # Sixth required model feature
            product_cost = st.number_input("Product Cost (costo_prod)", 
                                         value=round(unit_price*0.8, 2),
                                         help="Cost of producing/acquiring the product")
            # Fourth required model feature
            received_quantity = st.number_input("Quantity Received (cant_recibida)", 
                                             min_value=0, max_value=None, value=quantity,
                                             help="Actual quantity received (may differ from ordered)")
            # Fifth required model feature (calculated)
            received_amount = st.number_input("Amount Received (monto_recibido)",
                                           value=received_quantity*unit_price,
                                           help="Total amount received (received quantity × price)")
        
        # Delivery details
        st.subheader("Delivery Details")
        col1, col2 = st.columns(2)
        
        with col1:
            # Seventh required model feature
            reception_days = st.number_input("Reception Days", min_value=1, value=30,
                                          help="Days between order and reception")
            # Eighth required model feature (calculated)
            amount_difference = st.number_input("Amount Difference", 
                                             value=order_amount-received_amount,
                                             help="Difference between ordered and received amounts")
        
        with col2:
            # Tenth required model feature
            delivery_time_diff = st.number_input("Delivery Time Difference (days)", 
                                              value=0, min_value=-100, max_value=100,
                                              help="Difference between scheduled and actual delivery (+ = late, - = early)")
            # Ninth required model feature
            total_amount = st.number_input("Total Amount", 
                                         value=order_amount,
                                         help="Total amount of the order")
        
        # Auto-calculate option
        auto_calc = st.checkbox("Auto-calculate derived values", value=True)
        if auto_calc:
            order_amount = quantity * unit_price
            received_amount = received_quantity * unit_price
            amount_difference = order_amount - received_amount
            total_amount = order_amount
        
        submit_button = st.form_submit_button("Predict OTIF")
    
    if submit_button:
        # Final calculation of any derived values if auto-calc is enabled
        if auto_calc:
            order_amount = quantity * unit_price
            received_amount = received_quantity * unit_price
            amount_difference = order_amount - received_amount
            total_amount = order_amount
        
        # Create input dataframe with exactly the 10 required features
        input_data = pd.DataFrame({
            'cant_prod_odc': [quantity],
            'prec_unt': [unit_price],
            'monto_odc': [order_amount],
            'cant_recibida': [received_quantity],
            'monto_recibido': [received_amount],
            'costo_prod': [product_cost],
            'reception_days': [reception_days],
            'amount_difference': [amount_difference],
            'total_amount': [total_amount],
            'delivery_time_diff': [delivery_time_diff]
        })
        
        # Store reference info (not used by model)
        reference_info = {
            'supplier': selected_supplier,
            'product': selected_product,
            'category': selected_category
        }
        
        # Make prediction
        predictions, probabilities = predict_otif(model, input_data)
        
        if predictions is not None:
            st.markdown('<div class="dashboard-section-title">Prediction Results</div>', unsafe_allow_html=True)
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="dashboard-section-title" style="font-size: 1.2rem;">Prediction Result</div>', unsafe_allow_html=True)
                if predictions[0]:
                    st.markdown('<p class="prediction-positive">✅ OTIF - On Time In Full</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="prediction-negative">❌ Not OTIF</p>', unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="result-metric">
                    <strong>Probability:</strong> {probabilities[0]*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
                
                # Show key factors
                st.markdown('<div class="dashboard-section-title" style="font-size: 1.2rem;">Key Factors</div>', unsafe_allow_html=True)
                if quantity != received_quantity:
                    st.markdown(f"""
                    <div class="result-metric" style="border-color: rgba(255, 193, 7, 0.5); background-color: rgba(255, 193, 7, 0.1);">
                        <strong>Quantity discrepancy:</strong> Ordered {quantity} but received {received_quantity}
                    </div>
                    """, unsafe_allow_html=True)
                if delivery_time_diff != 0:
                    if delivery_time_diff > 0:
                        st.markdown(f"""
                        <div class="result-metric" style="border-color: rgba(220, 53, 69, 0.5); background-color: rgba(220, 53, 69, 0.1);">
                            <strong>Late delivery:</strong> {delivery_time_diff} days behind schedule
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-metric" style="border-color: rgba(25, 135, 84, 0.5); background-color: rgba(25, 135, 84, 0.1);">
                            <strong>Early delivery:</strong> {abs(delivery_time_diff)} days ahead of schedule
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="dashboard-section-title" style="font-size: 1.2rem;">Order Summary</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="order-summary">
                    <div style="margin-bottom: 10px;"><strong>Supplier:</strong> {selected_supplier}</div>
                    <div style="margin-bottom: 10px;"><strong>Product:</strong> {selected_product}</div>
                    <div style="margin-bottom: 10px;"><strong>Category:</strong> {selected_category}</div>
                    <div style="margin-bottom: 10px;"><strong>Order Amount:</strong> ${order_amount:.2f}</div>
                    <div style="margin-bottom: 10px;"><strong>Received Amount:</strong> ${received_amount:.2f}</div>
                    <div><strong>Reception Days:</strong> {reception_days}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show model input features with styled expander
            st.markdown('<div class="dashboard-section-title" style="font-size: 1.2rem;">Model Details</div>', unsafe_allow_html=True)
            with st.expander("View Model Input Features"):
                st.dataframe(input_data)
                
                st.markdown("""
                ### Feature Explanations
                These are the exact 10 features that were selected as most important by the ML model:
                
                - **cant_prod_odc**: The quantity of products ordered
                - **prec_unt**: Unit price of the product  
                - **monto_odc**: Total order amount (quantity × unit price)
                - **cant_recibida**: Actual quantity received
                - **monto_recibido**: Amount received (received quantity × unit price)
                - **costo_prod**: Cost of the product
                - **reception_days**: Days between order and reception
                - **amount_difference**: Difference between ordered and received amounts
                - **total_amount**: Total order amount 
                - **delivery_time_diff**: Difference between scheduled and actual delivery
                """)
                
            # Show similar past orders
            st.markdown('<div class="dashboard-section-title" style="font-size: 1.2rem;">Similar Past Orders</div>', unsafe_allow_html=True)
            similar_orders = data[
                (data['nom_prov'] == selected_supplier) | 
                (data['Categoria'] == selected_category)
            ].head(5)
            
            if not similar_orders.empty:
                st.dataframe(similar_orders[['fecha_odc', 'nom_prov', 'descrip_prod', 
                                           'cant_prod_odc', 'monto_odc', 'delivery_days', 'OTIF']])
            else:
                st.markdown("""
                <div style="background-color: rgba(26, 41, 128, 0.1); padding: 15px; border-radius: 8px; border: 1px solid rgba(38, 208, 206, 0.3);">
                    <div style="display: flex; align-items: center;">
                        <div style="font-size: 1.5rem; margin-right: 10px; color: #1a2980;">ℹ️</div>
                        <div>No similar orders found in the dataset.</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# Create supplier analysis section
def create_supplier_analysis(data):
    st.markdown('<div class="dashboard-section-title">Supplier Performance Analysis</div>', unsafe_allow_html=True)
    
    # Add mode-adaptive CSS with reliable radio button styling
    st.markdown("""
    <style>
        /* Section headers - ensure visibility in both modes */
        h3 {
            color: white !important;
            background: linear-gradient(to right, rgba(26, 41, 128, 0.8), rgba(38, 208, 206, 0.8));
            padding: 8px 15px;
            border-radius: 8px;
            font-size: 1.3rem;
            margin: 20px 0 15px 0;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
        }
        
        /* Metric cards - adaptive styling */
        .metric-card {
            background: linear-gradient(135deg, rgba(26, 41, 128, 0.1), rgba(38, 208, 206, 0.1));
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 0.15rem 0.5rem rgba(0, 0, 0, 0.1);
            text-align: center;
            margin-bottom: 1rem;
            border: 1px solid rgba(38, 208, 206, 0.3);
            backdrop-filter: blur(5px);
        }
        
        /* Metric values with gradient text */
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            background: linear-gradient(to right, #1a2980, #26d0ce);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.3rem;
            text-shadow: none;
        }
        
        /* Ensure metric labels are visible */
        .metric-label {
            font-size: 1rem;
            color: inherit !important;
            opacity: 0.9;
        }
        
        /* Radio button container styling */
        div.stRadio > div {
            background-color: transparent;
            padding: 0;
        }
        
        /* Radio button label styling */
        div.stRadio label {
            color: inherit !important;
        }
        
        /* Ensure dataframe headers are visible */
        .dataframe th {
            background-color: rgba(26, 41, 128, 0.2) !important;
            color: inherit !important;
            font-weight: bold !important;
        }
        
        /* Style dataframe cells */
        .dataframe td {
            background-color: rgba(255, 255, 255, 0.05) !important;
            color: inherit !important;
        }
        
        /* Custom supplier filters container */
        .supplier-filter-container {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 20px 0;
        }
        
        /* Filter option styling */
        .filter-option {
            background: linear-gradient(135deg, rgba(26, 41, 128, 0.1), rgba(38, 208, 206, 0.1));
            padding: 10px 15px;
            border-radius: 6px;
            border: 1px solid rgba(38, 208, 206, 0.3);
            cursor: pointer;
            transition: all 0.2s ease;
            color: inherit !important;
            font-weight: 500;
        }
        
        /* Filter option hover effect */
        .filter-option:hover {
            background: linear-gradient(135deg, rgba(26, 41, 128, 0.2), rgba(38, 208, 206, 0.2));
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        /* Active filter option */
        .filter-option.active {
            background: linear-gradient(135deg, rgba(26, 41, 128, 0.3), rgba(38, 208, 206, 0.3));
            border: 1px solid rgba(38, 208, 206, 0.8);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Group by supplier
    supplier_stats = data.groupby('nom_prov').agg({
        'OTIF': 'mean',
        'ontime': 'mean',
        'delivery_days': 'mean',
        'nro_odc': 'count',
        'monto_odc': 'sum'
    }).reset_index()
    
    supplier_stats = supplier_stats.rename(columns={
        'OTIF': 'OTIF Rate',
        'ontime': 'On-Time Rate',
        'delivery_days': 'Avg Delivery Days',
        'nro_odc': 'Number of Orders',
        'monto_odc': 'Total Order Amount'
    })
    
    # Convert rates to percentages
    supplier_stats['OTIF Rate'] = supplier_stats['OTIF Rate'] * 100
    supplier_stats['On-Time Rate'] = supplier_stats['On-Time Rate'] * 100
    
    # Use radio buttons instead of selectbox for sorting
    st.markdown("<div style='font-weight: 500; margin-bottom: 10px;'>Sort suppliers by:</div>", unsafe_allow_html=True)
    sort_by = st.radio(
        label="", 
        options=["OTIF Rate", "On-Time Rate", "Avg Delivery Days", "Number of Orders", "Total Order Amount"],
        horizontal=True
    )
    
    ascending = True if sort_by == "Avg Delivery Days" else False
    supplier_stats_sorted = supplier_stats.sort_values(by=sort_by, ascending=ascending)
    
    # Display top 10 suppliers
    st.markdown('<div class="dashboard-section-title" style="font-size: 1.3rem;">Top 10 Suppliers by {}</div>'.format(sort_by), unsafe_allow_html=True)
    
    # Add space for better visibility
    st.markdown("<br>", unsafe_allow_html=True)
    
    top_suppliers = supplier_stats_sorted.head(10)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='nom_prov', y=sort_by, data=top_suppliers, palette='viridis', ax=ax)
    ax.set_title(f'Top 10 Suppliers by {sort_by}', fontsize=16)
    ax.set_xlabel('Supplier', fontsize=14)
    ax.set_ylabel(sort_by, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show detailed supplier data
    st.markdown('<div class="dashboard-section-title" style="font-size: 1.3rem;">Detailed Supplier Performance</div>', unsafe_allow_html=True)
    st.dataframe(supplier_stats_sorted)
    
    # Supplier selection with radio buttons
    st.markdown("<div style='font-weight: 500; margin: 20px 0 10px 0;'>Select a supplier for detailed analysis:</div>", unsafe_allow_html=True)
    
    # Get the top 10 suppliers by the current sort metric
    top_supplier_names = top_suppliers['nom_prov'].tolist()
    
    # Use a radio button for supplier selection
    selected_supplier = st.radio(
        label="",
        options=top_supplier_names,
        horizontal=True
    )
    
    # Allow custom search if the supplier isn't in the top 10
    st.markdown("<div style='margin: 15px 0 5px 0;'>Or search for another supplier:</div>", unsafe_allow_html=True)
    
    # Create a text input for search
    other_supplier = st.text_input("", placeholder="Type a supplier name")
    
    # If user entered a search term, find matching suppliers
    if other_supplier:
        matches = [s for s in supplier_stats['nom_prov'].tolist() if other_supplier.lower() in s.lower()]
        if matches:
            selected_supplier = st.radio(
                "Select from matching suppliers:",
                options=matches,
                horizontal=len(matches) <= 5
            )
        else:
            st.warning("No suppliers found matching your search.")
    
    # Filter data for selected supplier
    supplier_data = data[data['nom_prov'] == selected_supplier]
    
    # Show supplier-specific metrics
    st.markdown('<div class="dashboard-section-title" style="font-size: 1.3rem;">Performance Metrics for {}</div>'.format(selected_supplier), unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        otif_rate = supplier_data['OTIF'].mean() * 100
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{otif_rate:.1f}%</div>
                <div class="metric-label">OTIF Rate</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        avg_days = supplier_data['delivery_days'].mean()
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{avg_days:.1f}</div>
                <div class="metric-label">Avg. Delivery Days</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col3:
        total_orders = len(supplier_data)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{total_orders}</div>
                <div class="metric-label">Total Orders</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Show supplier order history
    st.markdown('<div class="dashboard-section-title" style="font-size: 1.3rem;">Order History</div>', unsafe_allow_html=True)
    st.dataframe(supplier_data[['fecha_odc', 'descrip_prod', 'cant_prod_odc', 
                              'prec_unt', 'monto_odc', 'delivery_days', 'OTIF']])
    
    # Time series analysis of supplier performance
    if len(supplier_data) > 5:  # Only if there's enough data
        st.markdown('<div class="dashboard-section-title" style="font-size: 1.3rem;">Performance Over Time</div>', unsafe_allow_html=True)
        
        # Group by month
        supplier_data['year_month'] = supplier_data['fecha_odc'].dt.to_period('M')
        monthly_perf = supplier_data.groupby('year_month').agg({
            'OTIF': 'mean',
            'ontime': 'mean',
            'delivery_days': 'mean'
        }).reset_index()
        
        monthly_perf['year_month'] = monthly_perf['year_month'].astype(str)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(x='year_month', y='OTIF', data=monthly_perf, marker='o', ax=ax)
        ax.set_title(f'{selected_supplier} - OTIF Rate Over Time', fontsize=16)
        ax.set_xlabel('Month', fontsize=14)
        ax.set_ylabel('OTIF Rate', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)




# Create product category analysis
def create_product_analysis(data):
    st.markdown('<div class="dashboard-section-title">Product Category Analysis</div>', unsafe_allow_html=True)
    
    # Add mode-adaptive CSS with reliable radio button styling
    st.markdown("""
    <style>
        /* Section headers - ensure visibility in both modes */
        h3 {
            color: white !important;
            background: linear-gradient(to right, rgba(26, 41, 128, 0.8), rgba(38, 208, 206, 0.8));
            padding: 8px 15px;
            border-radius: 8px;
            font-size: 1.3rem;
            margin: 20px 0 15px 0;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
        }
        
        /* Metric cards - adaptive styling */
        .metric-card {
            background: linear-gradient(135deg, rgba(26, 41, 128, 0.1), rgba(38, 208, 206, 0.1));
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 0.15rem 0.5rem rgba(0, 0, 0, 0.1);
            text-align: center;
            margin-bottom: 1rem;
            border: 1px solid rgba(38, 208, 206, 0.3);
            backdrop-filter: blur(5px);
        }
        
        /* Metric values with gradient text */
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            background: linear-gradient(to right, #1a2980, #26d0ce);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.3rem;
            text-shadow: none;
        }
        
        /* Ensure metric labels are visible */
        .metric-label {
            font-size: 1rem;
            color: inherit !important;
            opacity: 0.9;
        }
        
        /* Radio button container styling */
        div.stRadio > div {
            background-color: transparent;
            padding: 0;
        }
        
        /* Radio button label styling */
        div.stRadio label {
            color: inherit !important;
        }
        
        /* Ensure dataframe headers are visible */
        .dataframe th {
            background-color: rgba(26, 41, 128, 0.2) !important;
            color: inherit !important;
            font-weight: bold !important;
        }
        
        /* Style dataframe cells */
        .dataframe td {
            background-color: rgba(255, 255, 255, 0.05) !important;
            color: inherit !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Group by category
    category_stats = data.groupby('Categoria').agg({
        'OTIF': 'mean',
        'delivery_days': 'mean',
        'monto_odc': ['sum', 'mean'],
        'nro_odc': 'count'
    }).reset_index()
    
    category_stats.columns = ['Category', 'OTIF Rate', 'Avg Delivery Days', 
                            'Total Order Amount', 'Avg Order Amount', 'Number of Orders']
    
    # Convert rates to percentages
    category_stats['OTIF Rate'] = category_stats['OTIF Rate'] * 100
    
    # Allow user to select analysis type with radio buttons
    st.markdown("<div style='font-weight: 500; margin-bottom: 10px;'>Select analysis metric:</div>", unsafe_allow_html=True)
    analysis_type = st.radio(
        label="",
        options=["OTIF Rate", "Average Delivery Days", "Total Order Amount", "Average Order Amount"],
        horizontal=True
    )
    
    if analysis_type == "OTIF Rate":
        y_col = 'OTIF Rate'
        title = 'OTIF Rate by Product Category'
        ylabel = 'OTIF Rate (%)'
    elif analysis_type == "Average Delivery Days":
        y_col = 'Avg Delivery Days'
        title = 'Average Delivery Days by Product Category'
        ylabel = 'Days'
    elif analysis_type == "Total Order Amount":
        y_col = 'Total Order Amount'
        title = 'Total Order Amount by Product Category'
        ylabel = 'Amount ($)'
    else:
        y_col = 'Avg Order Amount'
        title = 'Average Order Amount by Product Category'
        ylabel = 'Amount ($)'
    
    # Sort categories by the selected metric
    ascending = True if y_col == 'Avg Delivery Days' else False
    category_stats_sorted = category_stats.sort_values(by=y_col, ascending=ascending)
    
    # Display category visualization
    st.markdown(f'<div class="dashboard-section-title" style="font-size: 1.3rem;">Category Performance by {analysis_type}</div>', unsafe_allow_html=True)
    
    # Add space for better visibility
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Category', y=y_col, data=category_stats_sorted, palette='viridis', ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Category', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show detailed category data
    st.markdown('<div class="dashboard-section-title" style="font-size: 1.3rem;">Detailed Category Performance</div>', unsafe_allow_html=True)
    st.dataframe(category_stats_sorted)
    
    # Subcategory analysis with radio buttons
    st.markdown('<div class="dashboard-section-title" style="font-size: 1.3rem;">Subcategory Analysis</div>', unsafe_allow_html=True)
    
    # Get categories as a list
    categories = list(data['Categoria'].unique())
    
    # Category selection with radio buttons
    st.markdown("<div style='font-weight: 500; margin: 20px 0 10px 0;'>Select a category to explore subcategories:</div>", unsafe_allow_html=True)
    
    # Use a radio button grid for category selection if there are many categories
    if len(categories) > 6:
        # Display in 2 columns
        col1, col2 = st.columns(2)
        with col1:
            selected_category = st.radio(
                label="",
                options=categories[:len(categories)//2],
                key="cat_radio1"
            )
        with col2:
            cat_radio2 = st.radio(
                label="",
                options=categories[len(categories)//2:],
                key="cat_radio2"
            )
        
        # Determine which radio button was selected
        if 'cat_radio2' in st.session_state and st.session_state.cat_radio2 != categories[len(categories)//2]:
            selected_category = cat_radio2
    else:
        # If few categories, just use a single row of radio buttons
        selected_category = st.radio(
            label="",
            options=categories,
            horizontal=True if len(categories) <= 5 else False
        )
    
    # Filter data for selected category
    category_data = data[data['Categoria'] == selected_category]
    
    # Display category-specific metrics
    st.markdown(f'<div class="dashboard-section-title" style="font-size: 1.3rem;">Performance Metrics for {selected_category}</div>', unsafe_allow_html=True)
    
    # Display key metrics for the selected category in cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        otif_rate = category_data['OTIF'].mean() * 100
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{otif_rate:.1f}%</div>
                <div class="metric-label">OTIF Rate</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        avg_days = category_data['delivery_days'].mean()
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{avg_days:.1f}</div>
                <div class="metric-label">Avg. Delivery Days</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col3:
        num_orders = len(category_data)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{num_orders}</div>
                <div class="metric-label">Number of Orders</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Group by subcategory
    subcategory_stats = category_data.groupby('Subcategoria').agg({
        'OTIF': 'mean',
        'delivery_days': 'mean',
        'monto_odc': ['sum', 'mean'],
        'nro_odc': 'count'
    }).reset_index()
    
    subcategory_stats.columns = ['Subcategory', 'OTIF Rate', 'Avg Delivery Days', 
                               'Total Order Amount', 'Avg Order Amount', 'Number of Orders']
    
    # Convert rates to percentages
    subcategory_stats['OTIF Rate'] = subcategory_stats['OTIF Rate'] * 100
    
    # Sort subcategories by the same metric used for categories
    subcategory_stats_sorted = subcategory_stats.sort_values(by=y_col, ascending=ascending)
    
    # Create visualization
    st.markdown(f'<div class="dashboard-section-title" style="font-size: 1.3rem;">Subcategory Analysis for {selected_category}</div>', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Subcategory', y=y_col, data=subcategory_stats_sorted, palette='viridis', ax=ax)
    ax.set_title(f'{analysis_type} by Subcategory for {selected_category}', fontsize=16)
    ax.set_xlabel('Subcategory', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show detailed subcategory data
    st.markdown('<div class="dashboard-section-title" style="font-size: 1.3rem;">Detailed Subcategory Performance</div>', unsafe_allow_html=True)
    st.dataframe(subcategory_stats_sorted)




# Create a correlation analysis section
def create_correlation_analysis(data):
    #st.markdown('<div class="sub-header" style="color: white;">Correlation Analysis</div>', unsafe_allow_html=True)

    # Add custom CSS to ensure dropdowns are visible in all modes
    st.markdown("""
    <style>
        /* Force label text visibility */
        .stSelectbox label p, .stMultiSelect label p {
            color: inherit !important;
            font-weight: bold !important;
            font-size: 1rem !important;
        }
        
        /* Style the selectbox container */
        .stSelectbox > div > div[data-baseweb="select"] > div,
        .stMultiSelect > div > div[data-baseweb="select"] > div {
            background-color: white !important;
            border: 1px solid #ccc !important;
            color: #333 !important;
        }
        
        /* Style selected values in the dropdown box */
        .stMultiSelect div[role="button"] {
            color: #333 !important;
            background-color: white !important;
        }
        
        /* Style dropdown menu */
        div[role="listbox"] {
            background-color: white !important;
            color: #333 !important;
        }
        
        /* Style dropdown options */
        div[role="listbox"] ul li {
            color: #333 !important;
            background-color: white !important;
        }
        
        /* Hover effect for options */
        div[role="listbox"] ul li:hover {
            background-color: #f0f0f0 !important;
        }
        
        /* Force multiselect chip colors */
        .stMultiSelect div[data-testid="stVerticalBlock"] span {
            color: #333 !important;
            background-color: #e6f3ff !important;
            border: 1px solid #bbd9ff !important;
        }
        
        /* Style the placeholder text */
        .stMultiSelect [data-baseweb="tag"] span,
        .stMultiSelect [data-baseweb="select"] [aria-selected="true"] {
            color: #333 !important;
        }
        
        /* Fix the width of the select boxes */
        .stSelectbox, .stMultiSelect {
            width: 100%;
        }
        
        /* Fix clear button in multiselect */
        .stMultiSelect [role="button"] svg {
            fill: #333 !important;
        }
        
        /* Style dropdown arrow */
        [data-baseweb="select"] svg {
            fill: #333 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.write("This section shows correlations between different metrics to help identify factors that influence OTIF performance.")
    
    # Select only numeric columns for correlation
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Ensure default columns exist in the dataset
    default_cols = ['OTIF', 'ontime', 'delivery_days', 'cant_prod_odc']
    # Add more defaults if they exist
    if 'percentage_received' in numeric_cols:
        default_cols.append('percentage_received')
    if 'tmp_entrega' in numeric_cols:
        default_cols.append('tmp_entrega')
    
    # Filter defaults to only include columns that exist in the dataset
    valid_defaults = [col for col in default_cols if col in numeric_cols]
    
    # Let user select columns of interest
    selected_cols = st.multiselect(
        "Select metrics for correlation analysis:",
        options=numeric_cols,
        default=valid_defaults[:4]  # Use at most 4 defaults to avoid overwhelming the visualization
    )
    
    if len(selected_cols) < 2:
        st.warning("Please select at least 2 metrics for correlation analysis.")
    else:
        # Create correlation heatmap
        plot_heatmap(data, selected_cols, "Correlation Between Supply Chain Metrics")
        
        # NEW MULTI-SELECT SCATTER PLOT SECTION
        st.markdown('<div class="dashboard-section-title" style="font-size: 1.3rem;">Explore Relationships Between Metrics</div>', unsafe_allow_html=True)
        
        # Let users select multiple metrics for scatter plots
        scatter_cols = st.multiselect(
            "Select metrics to generate scatter plots (select at least 2):",
            options=selected_cols,
            default=selected_cols[:min(3, len(selected_cols))]
        )
        
        if len(scatter_cols) < 2:
            st.warning("Please select at least 2 metrics to generate scatter plots.")
        else:
            # Create scatter plot matrix for all selected combinations
            import itertools
            pairs = list(itertools.combinations(scatter_cols, 2))
            
            # Calculate how many rows we need (2 plots per row)
            num_pairs = len(pairs)
            num_rows = (num_pairs + 1) // 2
            
            # Create plots in a grid
            for i in range(num_rows):
                col1, col2 = st.columns(2)
                
                # First plot in the row
                if i*2 < num_pairs:
                    x_metric, y_metric = pairs[i*2]
                    with col1:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.scatterplot(x=x_metric, y=y_metric, data=data, hue='OTIF', palette=['red', 'green'], ax=ax)
                        ax.set_title(f'{x_metric} vs {y_metric}', fontsize=16)
                        ax.set_xlabel(x_metric, fontsize=14)
                        ax.set_ylabel(y_metric, fontsize=14)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Show correlation value
                        correlation = data[[x_metric, y_metric]].corr().iloc[0, 1]
                        st.markdown(f"**Correlation: {correlation:.3f}**")
                        
                        # Interpretation with styled colors
                        if abs(correlation) > 0.7:
                            st.markdown("<div class='correlation-strong'>Strong correlation</div>", unsafe_allow_html=True)
                        elif abs(correlation) > 0.4:
                            st.markdown("<div class='correlation-moderate'>Moderate correlation</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div class='correlation-weak'>Weak correlation</div>", unsafe_allow_html=True)
                
                # Second plot in the row
                if i*2 + 1 < num_pairs:
                    x_metric, y_metric = pairs[i*2 + 1]
                    with col2:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.scatterplot(x=x_metric, y=y_metric, data=data, hue='OTIF', palette=['red', 'green'], ax=ax)
                        ax.set_title(f'{x_metric} vs {y_metric}', fontsize=16)
                        ax.set_xlabel(x_metric, fontsize=14)
                        ax.set_ylabel(y_metric, fontsize=14)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Show correlation value
                        correlation = data[[x_metric, y_metric]].corr().iloc[0, 1]
                        st.markdown(f"**Correlation: {correlation:.3f}**")
                        
                        # Interpretation with styled colors
                        if abs(correlation) > 0.7:
                            st.markdown("<div class='correlation-strong'>Strong correlation</div>", unsafe_allow_html=True)
                        elif abs(correlation) > 0.4:
                            st.markdown("<div class='correlation-moderate'>Moderate correlation</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div class='correlation-weak'>Weak correlation</div>", unsafe_allow_html=True)

# Main function to run the Streamlit app
def main():
    # Load CSS
    load_css()
    
    # Load data
    data_file = os.path.join('app', 'cleaned_data.csv')
    data = load_data(data_file)
    
    if data is None:
        st.error("Could not load data. Please check the file path.")
        return
    
    # Load the trained ML model
    model = load_ml_model()
    
    # App header
    #st.markdown('<div class="main-header" style="color: white;">Supply Chain Analytics & Prediction Platform</div>', unsafe_allow_html=True)
    
    # Display the main image
    #display_main_image(os.path.join('app', 'banner.png'))
    
    # Call the stylish sidebar function to get the selected page
    selection = create_stylish_sidebar(data)
    
    # Page content based on selection
    if selection == "Project Overview":
        # Custom CSS for project overview page - fixed for both light and dark modes
        st.markdown("""
        <style>
            /* Card styles - adapted for both modes */
            .card {
                border: 1px solid rgba(128, 128, 128, 0.2);
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                background-color: rgba(255, 255, 255, 0.05);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            }
            
            /* Card title with gradient background - visible in both modes */
            .card-title {
                color: white !important; /* Always white text */
                font-size: 1.5rem;
                margin-bottom: 15px;
                border-bottom: 1px solid rgba(128, 128, 128, 0.2);
                padding: 8px 15px;
                border-radius: 8px;
                background: linear-gradient(to right, #1a2980, #26d0ce); /* Your gradient */
                text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            
            .tech-badge {
                display: inline-block;
                background-color: rgba(26, 41, 128, 0.1);
                color: inherit; /* Inherits text color from parent */
                padding: 5px 10px;
                border-radius: 15px;
                margin: 5px;
                font-size: 0.9rem;
                border: 1px solid rgba(26, 41, 128, 0.2);
            }
            
            .github-button {
                display: inline-flex;
                align-items: center;
                background-color: #333;
                color: white !important;
                padding: 10px 20px;
                border-radius: 5px;
                text-decoration: none;
                font-weight: bold;
                margin-top: 20px;
                transition: background-color 0.3s;
            }
            
            .github-button:hover {
                background-color: #2c3e50;
            }
            
            .phase-badge {
                display: inline-block;
                background-color: rgba(255, 255, 255, 0.1);
                color: inherit;
                padding: 6px 12px;
                border-radius: 6px;
                margin: 4px;
                font-size: 0.9rem;
                font-weight: 500;
                border: 1px solid rgba(52, 152, 219, 0.5);
            }
        </style>
        """, unsafe_allow_html=True)
        
        
        # Introduction text
        st.markdown("""
        This interactive platform provides comprehensive analysis of supply chain data using Python and Power BI. 
        It combines data visualization, statistical analysis and machine learning to deliver actionable insights
        for optimizing supply chain operations.
        """)
        
        # GitHub button
        st.markdown("""
        <a href="https://github.com/Jotis86/Supply-Chain-Project" class="github-button">
            <svg height="24" width="24" viewBox="0 0 16 16" version="1.1">
                <path fill="white" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
            </svg>
            &nbsp;View on GitHub
        </a>
        """, unsafe_allow_html=True)

        # Add space
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Objectives card
        #st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🎯 Project Objectives</div>', unsafe_allow_html=True)
        
        # Two columns for objectives
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            - 📊 **Interactive metrics analysis**
            - 📈 **KPI-driven decision support**
            - 🔍 **Pattern & trend identification**
            - 🏭 **Supplier performance tracking**
            """)
        
        with col2:
            st.markdown("""
            - 💡 **Data-driven decision making**
            - 🚀 **Supply chain optimization**
            - 🔮 **ML-powered delivery prediction**
            - ⚠️ **Proactive risk management**
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Add space
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Simplified Project timeline as badges
        #st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">⏱️ Project Timeline</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center;">
            <span class="phase-badge">1️⃣ Data Collection</span>
            <span class="phase-badge">2️⃣ EDA</span>
            <span class="phase-badge">3️⃣ Feature Engineering</span>
            <span class="phase-badge">4️⃣ Model Development</span>
            <span class="phase-badge">5️⃣ Dashboard Creation</span>
            <span class="phase-badge">6️⃣ Deployment</span>
        </div>
        
        <div style="margin-top: 15px;">
            The project followed a structured approach to data science, starting with data gathering and preparation, 
            proceeding through analysis and model development, and culminating in the creation of interactive dashboards 
            and deployment of prediction tools.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Add space
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Technologies used
        #st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🛠️ Technology Stack</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Data Processing:**
            <div>
                <span class="tech-badge">Python</span>
                <span class="tech-badge">Pandas</span>
                <span class="tech-badge">NumPy</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            **Visualization:**
            <div>
                <span class="tech-badge">Matplotlib</span>
                <span class="tech-badge">Seaborn</span>
                <span class="tech-badge">Power BI</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            **Machine Learning:**
            <div>
                <span class="tech-badge">Scikit-learn</span>
                <span class="tech-badge">Joblib</span>
                <span class="tech-badge">SMOTE</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        

    elif selection == "Dashboard":

        # Add CSS with gradient backgrounds for headers to ensure visibility in both modes
        st.markdown("""
        <style>
            /* Dashboard section headers - with gradient background for visibility in both modes */
            h2, h3, h4 {
                color: white !important;
                background: linear-gradient(to right, #1a2980, #26d0ce);
                padding: 8px 15px;
                border-radius: 8px;
                margin-top: 20px;
                margin-bottom: 20px;
                text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            
            /* KPI explanation container - adaptive styling */
            .kpi-explanation-container {
                margin-top: 10px;
                margin-bottom: 30px;
                background-color: rgba(26, 41, 128, 0.1);
                border-radius: 10px;
                padding: 15px;
                border: 1px solid rgba(52, 152, 219, 0.5);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            /* Individual KPI explanation cards */
            .kpi-explanation {
                display: flex;
                margin-bottom: 10px;
                padding: 10px;
                border-radius: 5px;
                background-color: rgba(26, 41, 128, 0.2);
                border: 1px solid rgba(52, 152, 219, 0.3);
            }
            
            /* Make sure KPI text is always visible */
            .kpi-explanation h4 {
                color: white !important;
                background: none;
                padding: 0;
                margin: 0 0 8px 0;
                box-shadow: none;
            }
            
            /* Ensure section titles are visible */
            .dashboard-section-title {
                color: white !important;
                background: linear-gradient(to right, #1a2980, #26d0ce);
                padding: 10px 15px;
                border-radius: 8px;
                font-size: 1.5rem;
                font-weight: bold;
                margin: 20px 0;
                text-align: center;
                text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
        </style>
        """, unsafe_allow_html=True)
        
        
        # Display key metrics
        display_key_metrics(data)

        # Add section title with new class
        st.markdown('<div class="dashboard-section-title">Key Supply Chain Metrics</div>', unsafe_allow_html=True)
        
        # Overview visualizations
        #col1, col2 = st.columns(2)


        # Container start
        #st.markdown('<div class="kpi-explanation-container">', unsafe_allow_html=True)

        # KPI 1
        st.markdown("""
            <div class="kpi-explanation">
                <div class="kpi-icon">📊</div>
                <div class="kpi-text">
                    <div class="kpi-title">OTIF Rate</div>
                    <div>Percentage of orders delivered both On Time and In Full. A key measure of overall supply chain effectiveness.</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # KPI 2
        st.markdown("""
            <div class="kpi-explanation">
                <div class="kpi-icon">🕒</div>
                <div class="kpi-text">
                    <div class="kpi-title">On-Time Delivery</div>
                    <div>Percentage of orders delivered by or before the promised delivery date, regardless of completeness.</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # KPI 3
        st.markdown("""
            <div class="kpi-explanation">
                <div class="kpi-icon">📆</div>
                <div class="kpi-text">
                    <div class="kpi-title">Average Delivery Days</div>
                    <div>The average number of days from order placement to delivery, a measure of supply chain speed.</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # KPI 4
        st.markdown("""
            <div class="kpi-explanation">
                <div class="kpi-icon">💰</div>
                <div class="kpi-text">
                    <div class="kpi-title">Cost Efficiency</div>
                    <div>Measures how efficiently resources are being utilized in the supply chain. Higher values indicate better cost optimization relative to order sizes.</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Container end
        st.markdown('</div>', unsafe_allow_html=True)

        # Distribution charts section
        st.markdown('<div class="dashboard-section-title">Distribution Analysis</div>', unsafe_allow_html=True)

        # Overview visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x='OTIF', data=data, palette=['#ff9999', '#66b3ff'])
            ax.set_title('OTIF Distribution', fontsize=16)
            ax.set_xlabel('On Time In Full', fontsize=14)
            ax.set_ylabel('Count', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x='ontime', data=data, palette=['#ff9999', '#66b3ff'])
            ax.set_title('On-Time Delivery Distribution', fontsize=16)
            ax.set_xlabel('On Time', fontsize=14)
            ax.set_ylabel('Count', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
        
        # Delivery days distribution
        st.markdown('<div class="dashboard-section-title">Delivery Time Analysis</div>', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(data['delivery_days'], kde=True, ax=ax, bins=30, color='skyblue')
        ax.set_title('Distribution of Delivery Days', fontsize=16)
        ax.set_xlabel('Delivery Days', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Top countries and categories
        col1, col2 = st.columns(2)
        
        with col1:
            # Top countries by order count
            country_counts = data['org_pais'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(10, 6))
            country_counts.plot(kind='bar', ax=ax, color=sns.color_palette('viridis', len(country_counts)))
            ax.set_title('Top 10 Countries by Order Count', fontsize=16)
            ax.set_xlabel('Country', fontsize=14)
            ax.set_ylabel('Count', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Top categories by order count
            category_counts = data['Categoria'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(10, 6))
            category_counts.plot(kind='bar', ax=ax, color=sns.color_palette('viridis', len(category_counts)))
            ax.set_title('Top 10 Categories by Order Count', fontsize=16)
            ax.set_xlabel('Category', fontsize=14)
            ax.set_ylabel('Count', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)

        # Create two columns for the efficiency visualizations
        # Add a section header for the order fulfillment analysis
        st.markdown('<div class="dashboard-section-title">Cost Efficiency Analysis</div>', unsafe_allow_html=True)

        # Create two columns for the cost efficiency visualizations
        col1, col2 = st.columns(2)

        with col1:
            # Cost per Unit by Category
            if 'monto_odc' in data.columns and 'cant_prod_odc' in data.columns and 'Categoria' in data.columns:
                # Calculate cost per unit
                plot_data = data.copy()
                plot_data['cost_per_unit'] = plot_data['monto_odc'] / plot_data['cant_prod_odc']
                
                # Remove outliers (greater than 95th percentile)
                upper_limit = np.percentile(plot_data['cost_per_unit'].dropna(), 95)
                plot_data = plot_data[plot_data['cost_per_unit'] <= upper_limit]
                
                # Group by category
                category_cost = plot_data.groupby('Categoria')['cost_per_unit'].mean().sort_values(ascending=False).head(10)
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = category_cost.plot(kind='bar', ax=ax, color=sns.color_palette('viridis', len(category_cost)))
                
                # Add data labels
                for i, bar in enumerate(ax.patches):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'${height:.2f}',
                        ha='center', va='bottom', rotation=0, fontsize=9)
                
                ax.set_title('Average Cost per Unit by Category', fontsize=16)
                ax.set_xlabel('Category', fontsize=14)
                ax.set_ylabel('Cost per Unit ($)', fontsize=14)
                plt.xticks(rotation=45, ha='right')
            else:
                # If columns don't exist, create a placeholder
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, 'Cost data not available', 
                        ha='center', va='center', fontsize=14)
                ax.set_xticks([])
                ax.set_yticks([])
            
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            # Order Size vs Cost Efficiency
            if 'monto_odc' in data.columns and 'cant_prod_odc' in data.columns:
                # Calculate cost per unit
                plot_data = data.copy()
                plot_data['cost_per_unit'] = plot_data['monto_odc'] / plot_data['cant_prod_odc']
                
                # Remove outliers
                upper_limit_cost = np.percentile(plot_data['cost_per_unit'].dropna(), 95)
                upper_limit_qty = np.percentile(plot_data['cant_prod_odc'].dropna(), 95)
                plot_data = plot_data[(plot_data['cost_per_unit'] <= upper_limit_cost) & 
                                    (plot_data['cant_prod_odc'] <= upper_limit_qty)]
                
                # Bin order quantities into groups
                plot_data['quantity_bin'] = pd.qcut(plot_data['cant_prod_odc'], 
                                                q=5, 
                                                labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'])
                
                # Group by quantity bin and calculate average cost per unit
                bin_costs = plot_data.groupby('quantity_bin')['cost_per_unit'].mean().reset_index()
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                order_categories = ['Very Small', 'Small', 'Medium', 'Large', 'Very Large']
                sorted_data = bin_costs.set_index('quantity_bin').reindex(order_categories).reset_index()
                
                bars = ax.bar(sorted_data['quantity_bin'], sorted_data['cost_per_unit'],
                        color=sns.color_palette('viridis', len(sorted_data)))
                
                # Add data labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'${height:.2f}',
                        ha='center', va='bottom', rotation=0, fontsize=9)
                
                ax.set_title('Cost Efficiency by Order Size', fontsize=16)
                ax.set_xlabel('Order Size', fontsize=14)
                ax.set_ylabel('Average Cost per Unit ($)', fontsize=14)
            else:
                # Create a placeholder if data is not available
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, 'Order size or cost data not available', 
                        ha='center', va='center', fontsize=14)
                ax.set_xticks([])
                ax.set_yticks([])
            
            plt.tight_layout()
            st.pyplot(fig)

    elif selection == "Power BI Dashboard":
        #st.markdown('<div class="dashboard-section-title">Power BI Dashboard</div>', unsafe_allow_html=True)
        st.write('''
        In this section, you can view screenshots of our Power BI dashboard. The dashboard provides comprehensive insights into the supply chain data with interactive visualizations and detailed reports.
        ''')

        # Display Power BI screenshots
        st.image(os.path.join('app', 'image_1.png'), caption='Purchases: General view of all purchases', use_container_width=True)
        st.image(os.path.join('app', 'image_2.png'), caption='Suppliers: Detailed analysis of suppliers', use_container_width=True)
        st.image(os.path.join('app', 'image_3.png'), caption='Products: Monitoring and analysis of the different products', use_container_width=True)

        display_video(os.path.join('app', 'video.mp4'))

    elif selection == "Correlation Analysis":
        create_correlation_analysis(data)

    elif selection == "Supplier Analysis":
        create_supplier_analysis(data)

    elif selection == "Product Analysis":
        create_product_analysis(data)

    elif selection == "OTIF Prediction":
        if model is None:
            st.error("ML model not found. Please ensure the model file exists.")
        else:
            create_prediction_form(data, model)

    # Footer
    st.markdown("""
    <div class="footer">
        Supply Chain Analytics & Prediction Platform | Developed by Jotis with love | &copy; 2025
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()