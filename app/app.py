import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Function to load data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Visualization functions
def plot_order_status_distribution(data):
    fig, ax = plt.subplots()
    sns.countplot(data=data, x='estado_odc', palette='viridis', ax=ax)
    ax.set_title('Order Status Distribution')
    ax.set_xlabel('Order Status')
    ax.set_ylabel('Count')
    st.pyplot(fig)

def plot_origin_country_distribution(data):
    fig, ax = plt.subplots()
    sns.countplot(data=data, x='org_pais', palette='magma', ax=ax)
    ax.set_title('Origin Country Distribution')
    ax.set_xlabel('Origin Country')
    ax.set_ylabel('Count')
    st.pyplot(fig)

def plot_unit_price_by_category(data):
    fig, ax = plt.subplots()
    sns.boxplot(data=data, x='Categoria', y='prec_unt', palette='coolwarm', ax=ax)
    ax.set_title('Unit Price by Category')
    ax.set_xlabel('Category')
    ax.set_ylabel('Unit Price')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

def plot_unit_price_by_subcategory(data):
    fig, ax = plt.subplots()
    sns.boxplot(data=data, x='Subcategoria', y='prec_unt', palette='coolwarm', ax=ax)
    ax.set_title('Unit Price by Subcategory')
    ax.set_xlabel('Subcategory')
    ax.set_ylabel('Unit Price')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

def plot_cost_vs_unit_price(data):
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x='costo_prod', y='prec_unt', color='darkblue', s=100, alpha=0.6, ax=ax)
    ax.set_title('Cost vs Unit Price')
    ax.set_xlabel('Cost')
    ax.set_ylabel('Unit Price')
    st.pyplot(fig)

def plot_delivery_days_vs_reception_days(data):
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x='delivery_days', y='reception_days', color='darkgreen', s=100, alpha=0.6, ax=ax)
    ax.set_title('Delivery Days vs Reception Days')
    ax.set_xlabel('Delivery Days')
    ax.set_ylabel('Reception Days')
    st.pyplot(fig)

def plot_total_amount_over_time(data):
    fig, ax = plt.subplots()
    sns.lineplot(data=data, x='fecha_odc', y='total_amount', color='coral', linewidth=2.5, ax=ax)
    ax.set_title('Total Amount Over Time')
    ax.set_xlabel('Order Date')
    ax.set_ylabel('Total Amount')
    st.pyplot(fig)

def plot_total_amount_by_category(data):
    category_sales = data.groupby('Categoria')['total_amount'].sum().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(data=category_sales, x='Categoria', y='total_amount', palette='Blues_d', ax=ax)
    ax.set_title('Total Amount by Category')
    ax.set_xlabel('Category')
    ax.set_ylabel('Total Amount')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

def plot_on_time_deliveries_by_provider(data):
    fig, ax = plt.subplots()
    sns.countplot(data=data, x='nom_prov', hue='ontime', palette='Set2', ax=ax)
    ax.set_title('On-time Deliveries by Provider')
    ax.set_xlabel('Provider')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=90)
    ax.legend(title='On-time')
    st.pyplot(fig)

def plot_otif_deliveries_by_provider(data):
    fig, ax = plt.subplots()
    sns.countplot(data=data, x='nom_prov', hue='OTIF', palette='Set3', ax=ax)
    ax.set_title('OTIF Deliveries by Provider')
    ax.set_xlabel('Provider')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=90)
    ax.legend(title='OTIF')
    st.pyplot(fig)

# Function to display the main image
def display_main_image(image_path):
    st.image(image_path, use_column_width=True)

# Function to display the video
def display_video(video_path):
    st.video(video_path)

# Main function to run the Streamlit app
def main():
    # Set the title of the app
    st.title('Supply Chain Dashboard')

    # Display the main image
    display_main_image('portada.png')

    # Load data
    data_file = 'data/cleaned_data.csv'
    data = load_data(data_file)

    # Sidebar menu for navigation
    st.sidebar.title('Navigation')
    menu = st.sidebar.radio('Go to', ['Introduction', 'Visualizations', 'Power BI Video'])

    if menu == 'Introduction':
        st.header('Introduction')
        st.write('This repository contains a comprehensive analysis of supply chain data using Power BI and Python. The project includes interactive visualizations, key metrics, and detailed reports to help in strategic decision making.')

    elif menu == 'Visualizations':
        st.header('Visualizations')
        st.write('## Data Preview')
        st.write(data.head())
        
        st.write('## Order Status Distribution')
        plot_order_status_distribution(data)
        
        st.write('## Origin Country Distribution')
        plot_origin_country_distribution(data)
        
        st.write('## Unit Price by Category')
        plot_unit_price_by_category(data)
        
        st.write('## Unit Price by Subcategory')
        plot_unit_price_by_subcategory(data)
        
        st.write('## Cost vs Unit Price')
        plot_cost_vs_unit_price(data)
        
        st.write('## Delivery Days vs Reception Days')
        plot_delivery_days_vs_reception_days(data)
        
        st.write('## Total Amount Over Time')
        plot_total_amount_over_time(data)
        
        st.write('## Total Amount by Category')
        plot_total_amount_by_category(data)
        
        st.write('## On-time Deliveries by Provider')
        plot_on_time_deliveries_by_provider(data)
        
        st.write('## OTIF Deliveries by Provider')
        plot_otif_deliveries_by_provider(data)

    elif menu == 'Power BI Video':
        st.header('Power BI Video')
        display_video('video.mp4')

if __name__ == '__main__':
    main()