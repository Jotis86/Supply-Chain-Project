# 📊 Supply Chain Project 📊

![Cover Image](assets/portada.png)

![Commits](https://img.shields.io/github/commit-activity/m/Jotis86/People-Analytics-Project)
![Issues Abiertas](https://img.shields.io/github/issues/Jotis86/People-Analytics-Project)
![Pull Requests](https://img.shields.io/github/issues-pr/Jotis86/People-Analytics-Project)
![Forks](https://img.shields.io/github/forks/Jotis86/People-Analytics-Project)
![Tamaño del Repositorio](https://img.shields.io/github/repo-size/Jotis86/People-Analytics-Project)
![Autor](https://img.shields.io/badge/autor-Juan%20Duran%20Bon-blue)
![Licencia](https://img.shields.io/github/license/Jotis86/People-Analytics-Project)

Welcome to the **Supply Chain** repository created with Power BI! 
This project aims to provide an interactive and detailed analysis of key metrics of a supply chain to support strategic decision making.

## 📋 Introduction

This repository contains a comprehensive analysis of supply chain data using Power BI and Python. The project includes interactive visualizations, key metrics, and detailed reports to help in strategic decision making.

## 🎯 Objectives

- 🎯 Provide an interactive and detailed analysis of supply chain metrics.
- 📊 Support strategic decision making with key performance indicators (KPIs).
- 📈 Identify patterns and trends over time.
- 📦 Analyze the performance of suppliers and products.

## ⚙️ Functionality

- 📈 Interactive visualizations with pivot charts and tables.
- 📊 Analysis of key metrics such as ontime, OTIF, delivery days, sales, and orders.
- 📅 Temporal analysis to identify patterns and opportunities.
- 📋 Detailed reports and dashboards.

## 🛠️ Tools Used

- **Power BI**: For creating interactive dashboards and visualizations.
- **Python**: For data extraction, transformation, and loading (ETL) processes.
- **Pandas**: For data manipulation and analysis.
- **Seaborn & Matplotlib**: For creating visualizations in Python.

## 🔄 Development Process

### ETL Process

- **Extraction**: Data obtained from Excel files.
- **Transformation**: 
  - 🗂️ Combining tables using Power Query.
  - 🧹 Data cleaning: Elimination of duplicates, treatment of null values, and data normalization.
  - 📈 Data enrichment: Aggregation of calculated columns and data transformation to improve analysis.
- **Load**: Integration of transformed data into Power BI for analysis and visualization.

### DAX Metrics

Various metrics have been created using **DAX (Data Analysis Expressions)** to provide detailed and customized analysis:
- 📊 **KPIs calculation**
- 📏 **Calculated measures**: Creation of custom measures for specific analyses.
- 📋 **Calculated columns**: Adding additional columns to enrich the data.
- 🔍 **Filtering and segmentation**: Use of DAX to apply filters and dynamic segmentations to the data.

## 📊 Results

### Power BI Dashboard

The Power BI dashboard includes:
- **Purchases**: General view of all purchases.
  ![Purchases](images/image_1.png)
- **Suppliers**: Detailed analysis of suppliers.
  ![Suppliers](images/image_2.png)
- **Products**: Monitoring and analysis of the different products.
  ![Products](images/image_3.png)

### Summary Report

This report provides a detailed analysis of the purchase orders and deliveries dataset. Below are the generated charts, the new calculated columns, and their added value to the analysis.

For a detailed analysis, please refer to the [Summary Report](summary_report/README.md).


## 📊 Visualizations


### Dashboard

A comprehensive dashboard with the following charts:
1. 📉 **Order Status Distribution**
2. 🌍 **Origin Country Distribution**
3. 💲 **Unit Price by Category**
4. 🏷️ **Unit Price by Subcategory**
5. 💵 **Cost vs Unit Price**
6. ⏳ **Delivery Days vs Reception Days**
7. 📅 **Total Amount Over Time**
8. 📊 **Total Amount by Category**
9. ⏰ **On-time Deliveries by Provider**
10. ✅ **OTIF Deliveries by Provider**

![Dashboard Image](assets/dashboard_2.png)

## 📂 Project Structure

- `SuplyChan.pbix`: Main file of the Power BI dashboard.
- `Purchases.xlxs`: Data from the marketing campaign used in the dashboard.
- `Images/`: Screenshots of the dashboard.
- `summary_report/`: Contains the summary report and visualizations.
- `data_visualization.ipynb`: Jupyter notebook with the data visualizations.

## 🌐 Web App (Streamlit)

A web application created using Streamlit to provide an interactive interface for the analysis.

## 🛠️ Requirements

- Power BI Desktop
- Python 3.x
- Pandas
- Seaborn
- Matplotlib
- Streamlit

## 📧 Contact

For any questions, you can contact me at:
- 📧 Email: jotaduranbon@gmail.com
- 💬 LinkedIn: [Your LinkedIn Profile](www.linkedin.com/in/juan-duran-bon)

## 💡 Suggestions and Contributions

Feel free to open issues or submit pull requests if you have any suggestions or contributions.

## 📜 License

This project is licensed under the [MIT License](LICENSE).

