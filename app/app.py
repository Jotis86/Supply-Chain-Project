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
        st.image(image_path, use_column_width=True)
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
    st.title('Supply Chain Dashboard')

    # Display the main image
    display_main_image(os.path.join('app','portada.png'))

    # Load data
    data_file = os.path.join('app', 'cleaned_data.csv')
    data = load_data(data_file)

    if data is not None:
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
            display_video('clip.mp4')

if __name__ == '__main__':
    main()