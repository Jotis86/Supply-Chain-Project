import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os

# Function to load data
def load_data(file_path):
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        return data
    else:
        st.error(f"Data file not found: {file_path}")
        return None

# Visualization functions
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
    
    # Remove x-tick labels
    ax.set_xticklabels([])
    
    st.pyplot(fig)

def plot_box(data, x_column, y_column, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, x=x_column, y=y_column, ax=ax, palette='coolwarm')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

def plot_bar(data, x_column, y_column, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=data, x=x_column, y=y_column, ax=ax, palette='Blues_d')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='x', rotation=90)
    st.pyplot(fig)

def plot_count(data, column, title, xlabel):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=data, x=column, ax=ax, palette='Set2')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.tick_params(axis='x', rotation=90)
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

# Main function to run the Streamlit app
def main():
    # Set the title of the app
    st.title('Supply Chain Project')

    # Display the main image
    display_main_image(os.path.join('app', 'portada.png'))

    # Load data
    data_file = os.path.join('app', 'cleaned_data.csv')
    data = load_data(data_file)

    # Sidebar menu for navigation
    st.sidebar.title('Navigation')
    st.sidebar.image(os.path.join('app', 'menu.png'), use_container_width=True)
    menu = st.sidebar.radio('Go to', ['Project Objectives', 'Development Process', 'Visualizations', 'Power BI Video', 'GitHub Repository'])

    if menu == 'Project Objectives':
        st.header('Introduction')
        st.write('''
        This repository contains a comprehensive analysis of supply chain data using Power BI and Python. The project includes interactive visualizations, key metrics, and detailed reports to help in strategic decision making.

        The supply chain is a critical component of any business, and understanding its dynamics can lead to significant improvements in efficiency and effectiveness. This project leverages the power of data analytics to provide insights into various aspects of the supply chain, from supplier performance to product trends.

        ### Objectives
        - 📊 **Provide an interactive and detailed analysis of supply chain metrics.**
        - 📈 **Support strategic decision making with key performance indicators (KPIs).**
        - 🔍 **Identify patterns and trends over time.**
        - 🏭 **Analyze the performance of suppliers and products.**
        - 💡 **Enhance data-driven decision making.**
        - 🚀 **Improve supply chain efficiency and effectiveness.**
        - 🌐 **Facilitate better understanding of supply chain dynamics.**
        - ⚠️ **Enable proactive management of supply chain risks.**
        - 🔄 **Foster continuous improvement in supply chain processes.**
        - 📋 **Deliver actionable insights for strategic planning.**

        ### Key Features
        - 🔗 **Data Integration**: Combining data from multiple sources to provide a holistic view of the supply chain.
        - 📊 **Interactive Dashboards**: Visualizations that allow users to explore data and gain insights.
        - 🔮 **Predictive Analytics**: Using historical data to forecast future trends and identify potential issues.
        - 📈 **Performance Metrics**: Tracking key performance indicators to measure the effectiveness of the supply chain.
        - 📝 **Custom Reports**: Generating detailed reports tailored to the needs of different stakeholders.

        ### Benefits
        - 🧠 **Improved Decision Making**: Access to accurate and timely information helps in making informed decisions.
        - 💸 **Increased Efficiency**: Identifying bottlenecks and inefficiencies in the supply chain can lead to significant cost savings.
        - 🤝 **Enhanced Collaboration**: Sharing insights with suppliers and partners fosters better collaboration and coordination.
        - 🛡️ **Risk Mitigation**: Proactively managing risks helps in avoiding disruptions and maintaining smooth operations.
        - 📈 **Strategic Planning**: Data-driven insights support long-term strategic planning and growth.

        We hope this project provides valuable insights and helps in optimizing your supply chain operations. 🚚
        ''')

    elif menu == 'Development Process':
        st.header('Development Process')
        st.write('''
        ### ETL Process
        The ETL (Extract, Transform, Load) process is a crucial part of our data pipeline. Here's a detailed breakdown of each step:

        - **Extraction**: 📥 Data obtained from various sources, primarily Excel files. This step involves gathering all necessary data for analysis.
        - **Transformation**: 🔄 This step involves several sub-processes:
        - **Combining Tables**: 🔗 Using Power Query to merge multiple tables into a single dataset.
        - **Data Cleaning**: 🧹 Elimination of duplicates, treatment of null values, and normalization of data to ensure consistency and accuracy.
        - **Data Enrichment**: 📈 Aggregation of calculated columns and transformation of data to enhance its value for analysis.
        - **Load**: 🚀 Integration of the transformed data into Power BI for analysis and visualization. This step ensures that the data is ready for reporting and insights generation.

        ### DAX Metrics
        DAX (Data Analysis Expressions) is a powerful formula language used in Power BI for data modeling. We have created various metrics using DAX to provide detailed and customized analysis:

        - **KPIs Calculation**: 📊 Key Performance Indicators (KPIs) are calculated to track the performance of different aspects of the supply chain.
        - **Calculated Measures**: 📐 Creation of custom measures for specific analyses, allowing for more precise and tailored insights.
        - **Calculated Columns**: 📏 Adding additional columns to enrich the data and provide more context for analysis.
        - **Filtering and Segmentation**: 🔍 Use of DAX to apply dynamic filters and segmentations to the data, enabling more granular analysis.

        ### Python Analysis
        In addition to Power BI, we have utilized Python for data analysis and visualization. Here are the steps involved:

        - **Data Loading**: 📂 Using pandas to load and manipulate the data.
        - **Data Visualization**: 📊 Creating various plots and charts using seaborn and matplotlib to gain insights into the data.
        - **Statistical Analysis**: 📈 Performing statistical analysis to identify trends and patterns in the data.

        ### Streamlit Application
        To present our findings interactively, we have developed a Streamlit application. This application allows users to explore the data and visualizations in an intuitive and user-friendly manner:

        - **Interactive Dashboards**: 📊 Users can interact with the visualizations to gain deeper insights.
        - **Real-time Updates**: 🔄 The application updates in real-time as users interact with it.
        - **User-friendly Interface**: 🖥️ The interface is designed to be easy to use, even for non-technical users.

        ### Additional Steps
        - **Data Validation**: ✅ Ensuring the accuracy and reliability of the data through rigorous validation checks.
        - **Automation**: 🤖 Automating repetitive tasks to improve efficiency and reduce the risk of human error.
        - **Documentation**: 📝 Maintaining comprehensive documentation of the ETL process, DAX metrics, Python analysis, and Streamlit application to ensure transparency and reproducibility.
        - **Collaboration**: 🤝 Working closely with stakeholders to understand their requirements and tailor the analysis to meet their needs.

        This comprehensive development process ensures that our data is accurate, reliable, and ready for insightful analysis. We hope this detailed explanation provides a clear understanding of the steps involved in our project. 📈🔍
        ''')

    elif menu == 'Visualizations':
        st.header('Visualizations')
        st.write('## Data Preview')
        st.write(data.head())

        # Visualization menu
        visualization_menu = st.selectbox('Select a visualization', [
            'Histogram of ordered product quantity',
            'Histogram of unit price',
            'Histogram of order amount',
            'Histogram of received quantity',
            'Pie chart of order status',
            'Line plot of order amount over time',
            'Line plot of received amount over time',
            'Box plot of unit price by order status',
            'Box plot of order amount by order status',
            'Bar plot of on-time deliveries',
            'Bar plot of OTIF deliveries',
            'Bar plot of origin country',
            'Box plot of unit price by category',
            'Box plot of unit price by subcategory',
            'Bar plot of on-time deliveries by provider',
            'Bar plot of OTIF deliveries by provider'
        ])

        if visualization_menu == 'Histogram of ordered product quantity':
            plot_histogram(data, 'cant_prod_odc', 'Histogram of Ordered Product Quantity', 'Ordered Product Quantity')

        elif visualization_menu == 'Histogram of unit price':
            plot_histogram(data, 'prec_unt', 'Histogram of Unit Price', 'Unit Price')

        elif visualization_menu == 'Histogram of order amount':
            plot_histogram(data, 'monto_odc', 'Histogram of Order Amount', 'Order Amount')

        elif visualization_menu == 'Histogram of received quantity':
            plot_histogram(data, 'cant_recibida', 'Histogram of Received Quantity', 'Received Quantity')

        elif visualization_menu == 'Pie chart of order status':
            plot_pie_chart(data, 'estado_odc', 'Pie Chart of Order Status')

        elif visualization_menu == 'Line plot of order amount over time':
            plot_line(data, 'fecha_odc', 'monto_odc', 'Line Plot of Order Amount Over Time', 'Order Date', 'Order Amount')

        elif visualization_menu == 'Line plot of received amount over time':
            plot_line(data, 'fecha_recibido', 'monto_recibido', 'Line Plot of Received Amount Over Time', 'Received Date', 'Received Amount')

        elif visualization_menu == 'Box plot of unit price by order status':
            plot_box(data, 'estado_odc', 'prec_unt', 'Box Plot of Unit Price by Order Status', 'Order Status', 'Unit Price')

        elif visualization_menu == 'Box plot of order amount by order status':
            plot_box(data, 'estado_odc', 'monto_odc', 'Box Plot of Order Amount by Order Status', 'Order Status', 'Order Amount')

        elif visualization_menu == 'Bar plot of on-time deliveries':
            plot_count(data, 'ontime', 'Bar Plot of On-time Deliveries', 'On-time')

        elif visualization_menu == 'Bar plot of OTIF deliveries':
            plot_count(data, 'OTIF', 'Bar Plot of OTIF Deliveries', 'OTIF')

        elif visualization_menu == 'Bar plot of origin country':
            plot_count(data, 'org_pais', 'Bar Plot of Origin Country', 'Origin Country')

        elif visualization_menu == 'Box plot of unit price by category':
            plot_box(data, 'Categoria', 'prec_unt', 'Box Plot of Unit Price by Category', 'Category', 'Unit Price')

        elif visualization_menu == 'Box plot of unit price by subcategory':
            plot_box(data, 'Subcategoria', 'prec_unt', 'Box Plot of Unit Price by Subcategory', 'Subcategory', 'Unit Price')

        elif visualization_menu == 'Bar plot of on-time deliveries by provider':
            plot_count(data, 'nom_prov', 'Bar Plot of On-time Deliveries by Provider', 'Provider')

        elif visualization_menu == 'Bar plot of OTIF deliveries by provider':
            plot_count(data, 'nom_prov', 'Bar Plot of OTIF Deliveries by Provider', 'Provider')

    elif menu == 'Power BI Video':
        st.header('Power BI Video')
        display_video(os.path.join('app', 'video.mp4'))

    elif menu == 'GitHub Repository':
        st.header('GitHub Repository')
        st.write('Visit the GitHub repository for more details and to access the source code.')
        if st.button('Go to GitHub Repo'):
            st.markdown('[GitHub Repository](https://github.com/Jotis86/Supply-Chain-Project)')


if __name__ == '__main__':
    main()