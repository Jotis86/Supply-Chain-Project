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


def create_stylish_sidebar(data):
    st.sidebar.markdown("""
    <style>
        /* Overall sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #2c3e50;
            color: white;
        }
        
        /* Sidebar header */
        .sidebar-header {
            text-align: center;
            font-size: 1.8rem;
            color: white;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
            font-weight: bold;
        }
        
        /* Navigation container */
        div.stRadio > div {
            background-color: transparent;
            padding: 0;
        }
        
        /* Navigation items */
        div.stRadio label {
            background-color: #34495e;
            color: white !important;
            padding: 12px 15px;
            border-radius: 6px;
            margin: 8px 0;
            display: flex;
            align-items: center;
            font-weight: 400;
            transition: all 0.2s;
        }
        
        /* Hover state */
        div.stRadio label:hover {
            background-color: #3498db;
            transform: translateX(5px);
        }
        
        /* Selected state */
        div.stRadio label[data-baseweb="radio"] input:checked + div {
            background-color: #3498db !important;
            border-color: white !important;
        }
        
        div.stRadio label[data-baseweb="radio"] input:checked + div + span {
            color: white !important;
            font-weight: bold !important;
        }
        
        /* Metrics container */
        .metrics-container {
            background-color: #34495e;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Individual metrics */
        .metric-mini {
            text-align: center;
            margin-bottom: 10px;
            padding: 5px;
            background-color: #2c3e50;
            border-radius: 6px;
        }
        
        .metric-mini-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #3498db;
        }
        
        .metric-mini-label {
            font-size: 0.85rem;
            color: white;
            margin-top: 4px;
        }
        
        /* Footer */
        .sidebar-footer {
            text-align: center;
            margin-top: 20px;
            padding-top: 10px;
            border-top: 1px solid #34495e;
            font-size: 0.8rem;
            color: #ecf0f1;
        }
        
        /* Make category headers more visible */
        .stRadio + div > label {
            color: white !important;
            font-weight: bold !important;
            font-size: 1.2rem !important;
            margin-top: 1rem;
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
        "📊 Dashboard",
        "🔮 OTIF Prediction",
        "🏭 Supplier Analysis",
        "📦 Product Analysis",
        "🔗 Correlation Analysis",
        "📋 Project Overview",
        "💼 Power BI Dashboard"
    ]
    
    selection = st.sidebar.radio("Navigation", pages_with_icons, label_visibility="collapsed")
    
    # Remove the icon to get the actual page name
    page = selection.split(" ", 1)[1]
    
    # Add footer
    st.sidebar.markdown("""
    <div class="sidebar-footer">
        Supply Chain Analytics Platform v1.0<br>
        &copy; 2023 All Rights Reserved
    </div>
    """, unsafe_allow_html=True)
    
    return page



# Add custom CSS
def load_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #2c3e50;
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
    st.markdown('<div class="sub-header">Key Supply Chain Metrics</div>', unsafe_allow_html=True)
    
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
        fulfilled_percentage = (data['cant_recibida'].sum() / data['cant_prod_odc'].sum() * 100)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{fulfilled_percentage:.1f}%</div>
                <div class="metric-label">Fulfillment Rate</div>
            </div>
            """, 
            unsafe_allow_html=True
        )

# Create a prediction form
def create_prediction_form(data, model):
    st.markdown('<div class="sub-header">OTIF Prediction Tool</div>', unsafe_allow_html=True)
    
    # Use Streamlit's native info box
    st.info("This tool predicts whether a delivery will be On Time In Full (OTIF) based on order parameters. Fill in the form below to generate a prediction.")
    
    # Get unique values for categorical fields
    suppliers = sorted(data['nom_prov'].unique())
    countries = sorted(data['org_pais'].unique())
    categories = sorted(data['Categoria'].unique())
    subcategories = sorted(data['Subcategoria'].unique())
    product_units = sorted(data['und_prod'].unique())
    products = sorted(data['descrip_prod'].unique())
    
    with st.form("prediction_form"):
        # Basic order information (category 1)
        st.subheader("Basic Order Information")
        col1, col2 = st.columns(2)
        
        with col1:
            selected_supplier = st.selectbox("Supplier", suppliers)
            selected_product = st.selectbox("Product", products)
            # First required model feature
            quantity = st.number_input("Quantity Ordered (cant_prod_odc)", min_value=1, value=100)
            # Second required model feature
            unit_price = st.number_input("Unit Price (prec_unt)", min_value=0.1, value=50.0, step=0.1)
            
        with col2:
            selected_category = st.selectbox("Category", categories)
            selected_unit = st.selectbox("Unit", product_units)
            # Third required model feature (calculated)
            order_amount = st.number_input("Order Amount (monto_odc)", 
                                        value=quantity*unit_price, 
                                        help="Total order amount (quantity × price)")
            # Sixth required model feature
            product_cost = st.number_input("Product Cost (costo_prod)", 
                                         value=round(unit_price*0.8, 2),
                                         help="Cost of producing/acquiring the product")
        
        # Delivery details (category 2)
        st.subheader("Delivery Details")
        col1, col2 = st.columns(2)
        
        with col1:
            # Fourth required model feature
            received_quantity = st.number_input("Quantity Received (cant_recibida)", 
                                             min_value=0, max_value=None, value=quantity,
                                             help="Actual quantity received (may differ from ordered)")
            # Fifth required model feature (calculated)
            received_amount = st.number_input("Amount Received (monto_recibido)",
                                           value=received_quantity*unit_price,
                                           help="Total amount received (received quantity × price)")
        
        with col2:
            # Seventh required model feature
            reception_days = st.number_input("Reception Days", min_value=1, value=30,
                                          help="Days between order and reception")
            # Tenth required model feature
            delivery_time_diff = st.number_input("Delivery Time Difference (days)", 
                                              value=0, min_value=-100, max_value=100,
                                              help="Difference between scheduled and actual delivery (+ = late, - = early)")
        
        # Financial details (category 3)
        st.subheader("Financial Details")
        col1, col2 = st.columns(2)
        
        with col1:
            # Eighth required model feature (calculated)
            amount_difference = st.number_input("Amount Difference", 
                                             value=order_amount-received_amount,
                                             help="Difference between ordered and received amounts")
        
        with col2:
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
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Prediction Result")
                if predictions[0]:
                    st.markdown('<p class="prediction-positive">✅ OTIF - On Time In Full</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="prediction-negative">❌ Not OTIF</p>', unsafe_allow_html=True)
                
                st.markdown(f"**Probability:** {probabilities[0]*100:.1f}%")
                
                # Show key factors
                st.markdown("### Key Factors")
                if quantity != received_quantity:
                    st.warning(f"Quantity discrepancy: Ordered {quantity} but received {received_quantity}")
                if delivery_time_diff != 0:
                    if delivery_time_diff > 0:
                        st.warning(f"Late delivery: {delivery_time_diff} days behind schedule")
                    else:
                        st.info(f"Early delivery: {abs(delivery_time_diff)} days ahead of schedule")
            
            with col2:
                st.markdown("### Order Summary")
                st.markdown(f"**Supplier:** {selected_supplier}")
                st.markdown(f"**Product:** {selected_product}")
                st.markdown(f"**Category:** {selected_category}")
                st.markdown(f"**Order Amount:** ${order_amount:.2f}")
                st.markdown(f"**Received Amount:** ${received_amount:.2f}")
                st.markdown(f"**Reception Days:** {reception_days}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show model input features
            with st.expander("Model Input Features"):
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
            st.markdown("### Similar Past Orders")
            similar_orders = data[
                (data['nom_prov'] == selected_supplier) | 
                (data['Categoria'] == selected_category)
            ].head(5)
            
            if not similar_orders.empty:
                st.dataframe(similar_orders[['fecha_odc', 'nom_prov', 'descrip_prod', 
                                           'cant_prod_odc', 'monto_odc', 'delivery_days', 'OTIF']])
            else:
                st.info("No similar orders found in the dataset.")


# Create supplier analysis section
def create_supplier_analysis(data):
    st.markdown('<div class="sub-header">Supplier Performance Analysis</div>', unsafe_allow_html=True)
    
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
    
    # Sort by different metrics
    sort_by = st.selectbox(
        "Sort suppliers by:", 
        ["OTIF Rate", "On-Time Rate", "Avg Delivery Days", "Number of Orders", "Total Order Amount"],
        index=0
    )
    
    ascending = True if sort_by == "Avg Delivery Days" else False
    supplier_stats_sorted = supplier_stats.sort_values(by=sort_by, ascending=ascending)
    
    # Display top 10 suppliers
    st.markdown(f"### Top 10 Suppliers by {sort_by}")
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
    st.markdown("### Detailed Supplier Performance")
    st.dataframe(supplier_stats_sorted)
    
    # Allow user to select a specific supplier for detailed analysis
    selected_supplier = st.selectbox("Select a supplier for detailed analysis:", 
                                     supplier_stats['nom_prov'].tolist())
    
    # Filter data for selected supplier
    supplier_data = data[data['nom_prov'] == selected_supplier]
    
    # Show supplier-specific metrics
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
    st.markdown("### Order History")
    st.dataframe(supplier_data[['fecha_odc', 'descrip_prod', 'cant_prod_odc', 
                              'prec_unt', 'monto_odc', 'delivery_days', 'OTIF']])
    
    # Time series analysis of supplier performance
    if len(supplier_data) > 5:  # Only if there's enough data
        st.markdown("### Performance Over Time")
        
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
    st.markdown('<div class="sub-header">Product Category Analysis</div>', unsafe_allow_html=True)
    
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
    
    # Display category metrics
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Category', y='OTIF Rate', data=category_stats, palette='viridis', ax=ax)
    ax.set_title('OTIF Rate by Product Category', fontsize=16)
    ax.set_xlabel('Category', fontsize=14)
    ax.set_ylabel('OTIF Rate (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Allow user to select analysis type
    analysis_type = st.selectbox(
        "Select category analysis:", 
        ["OTIF Rate", "Average Delivery Days", "Total Order Amount", "Average Order Amount"]
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
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Category', y=y_col, data=category_stats, palette='viridis', ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Category', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show detailed category data
    st.markdown("### Detailed Category Performance")
    st.dataframe(category_stats)
    
    # Subcategory analysis
    st.markdown("### Subcategory Analysis")
    
    # Select category for subcategory analysis
    selected_category = st.selectbox("Select a category for subcategory analysis:", 
                                    data['Categoria'].unique())
    
    # Filter data for selected category
    category_data = data[data['Categoria'] == selected_category]
    
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
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Subcategory', y='OTIF Rate', data=subcategory_stats, palette='viridis', ax=ax)
    ax.set_title(f'OTIF Rate by Subcategory for {selected_category}', fontsize=16)
    ax.set_xlabel('Subcategory', fontsize=14)
    ax.set_ylabel('OTIF Rate (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show detailed subcategory data
    st.dataframe(subcategory_stats)

# Create a correlation analysis section
def create_correlation_analysis(data):
    st.markdown('<div class="sub-header">Correlation Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    This section shows correlations between different metrics to help identify factors that influence OTIF performance.
    </div>
    """, unsafe_allow_html=True)
    
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
        
        # Scatter plot for exploring relationships
        st.markdown("### Explore Relationship Between Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_metric = st.selectbox("Select X-axis metric:", selected_cols, index=0)
        
        with col2:
            y_metric = st.selectbox("Select Y-axis metric:", selected_cols, index=min(1, len(selected_cols)-1))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=x_metric, y=y_metric, data=data, hue='OTIF', palette=['red', 'green'], ax=ax)
        ax.set_title(f'Relationship Between {x_metric} and {y_metric}', fontsize=16)
        ax.set_xlabel(x_metric, fontsize=14)
        ax.set_ylabel(y_metric, fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show correlation value
        correlation = data[[x_metric, y_metric]].corr().iloc[0, 1]
        st.markdown(f"**Correlation coefficient between {x_metric} and {y_metric}:** {correlation:.3f}")
        
        if abs(correlation) > 0.7:
            st.markdown("**Interpretation:** Strong correlation")
        elif abs(correlation) > 0.4:
            st.markdown("**Interpretation:** Moderate correlation")
        else:
            st.markdown("**Interpretation:** Weak correlation")

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
    st.markdown('<div class="main-header">Supply Chain Analytics & Prediction Platform</div>', unsafe_allow_html=True)
    
    # Display the main image
    display_main_image(os.path.join('app', 'banner.png'))
    
    # Call the stylish sidebar function to get the selected page
    selection = create_stylish_sidebar(data)
    
    # Page content based on selection
    if selection == "Dashboard":
        # Display key metrics
        display_key_metrics(data)
        
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
        st.markdown('<div class="section-header">Delivery Time Analysis</div>', unsafe_allow_html=True)
        
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
    
    elif selection == "OTIF Prediction":
        if model is None:
            st.error("ML model not found. Please ensure the model file exists.")
        else:
            create_prediction_form(data, model)
    
    elif selection == "Supplier Analysis":
        create_supplier_analysis(data)
    
    elif selection == "Product Analysis":
        create_product_analysis(data)
    
    elif selection == "Correlation Analysis":
        create_correlation_analysis(data)

    elif selection == "Project Overview":
        st.header('Project Objectives')
        
        st.write('''
        This project contains a comprehensive analysis of supply chain data using Power BI and Python. The project includes interactive visualizations, key metrics, detailed reports, and a machine learning model to predict OTIF (On Time In Full) delivery status.

        ### Objectives
        - 📊 **Provide an interactive and detailed analysis of supply chain metrics.**
        - 📈 **Support strategic decision making with key performance indicators (KPIs).**
        - 🔍 **Identify patterns and trends over time.**
        - 🏭 **Analyze the performance of suppliers and products.**
        - 💡 **Enhance data-driven decision making.**
        - 🚀 **Improve supply chain efficiency and effectiveness.**
        - 🔮 **Predict future delivery performance using machine learning.**
        - ⚠️ **Enable proactive management of supply chain risks.**
        - 🔄 **Foster continuous improvement in supply chain processes.**
        ''')
        
        st.header('Development Process')
        st.write('''
        ### ETL Process
        - **Extraction**: 📥 Data obtained from various sources, primarily Excel files.
        - **Transformation**: 🔄 Data cleaning, normalization, and feature engineering.
        - **Load**: 🚀 Integration of the transformed data for analysis and visualization.

        ### Machine Learning Model
        - **Data Preparation**: Feature selection and preprocessing.
        - **Model Selection**: Comparison of multiple classification algorithms.
        - **Training & Validation**: Model training with cross-validation.
        - **Deployment**: Implementation of the model in this interactive application.
        ''')
    
    elif selection == "Power BI Dashboard":
        st.header('Power BI Dashboard')
        st.write('''
        In this section, you can view screenshots of our Power BI dashboard. The dashboard provides comprehensive insights into the supply chain data with interactive visualizations and detailed reports.
        ''')

        # Display Power BI screenshots
        st.image(os.path.join('app', 'image_1.png'), caption='Purchases: General view of all purchases', use_container_width=True)
        st.image(os.path.join('app', 'image_2.png'), caption='Suppliers: Detailed analysis of suppliers', use_container_width=True)
        st.image(os.path.join('app', 'image_3.png'), caption='Products: Monitoring and analysis of the different products', use_container_width=True)

        display_video(os.path.join('app', 'video.mp4'))

    # Footer
    st.markdown("""
    <div class="footer">
        Supply Chain Analytics & Prediction Platform | Developed by Jotis with love | &copy; 2025
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()