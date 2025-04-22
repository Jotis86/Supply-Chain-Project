# 📦 Supply Chain Analytics & Prediction Platform

![Cover Image](app/banner.png)

![Commits](https://img.shields.io/github/commit-activity/m/Jotis86/Supply-Chain-Project)
![Issues Abiertas](https://img.shields.io/github/issues/Jotis86/Supply-Chain-Project)
![Pull Requests](https://img.shields.io/github/issues-pr/Jotis86/Supply-Chain-Project)
![Forks](https://img.shields.io/github/forks/Jotis86/Supply-Chain-Project)
![Tamaño del Repositorio](https://img.shields.io/github/repo-size/Jotis86/Supply-Chain-Project)
![Autor](https://img.shields.io/badge/autor-Juan%20Duran%20Bon-blue)
![Licencia](https://img.shields.io/github/license/Jotis86/Supply-Chain-Project)

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

## 🤖 Machine Learning Integration

The project now incorporates an advanced machine learning model to predict OTIF (On Time In Full) delivery outcomes:

- **OTIF Prediction Tool**: Interactive interface to forecast delivery success based on order parameters
- **Category-Filtered Selection**: Smart product selection based on chosen category
- **Real-time Predictions**: Immediate results with probability scores
- **Influential Factors**: Automatic identification of key variables affecting prediction outcomes
- **Similar Orders Analysis**: Contextual information through comparison with historical data
- **User-Friendly Interface**: Intuitive form design with clear explanations

## 📈 Advanced Analytics

### KPI Framework

The platform now includes a comprehensive KPI framework with detailed metrics:

- **OTIF Rate**: Percentage of orders delivered both On Time and In Full
- **On-Time Delivery**: Percentage of orders delivered by or before the promised date
- **Average Delivery Days**: Mean time from order to delivery
- **Cost Efficiency**: Resource utilization optimization metric with benchmark comparison

### Efficiency Analysis

New analytical capabilities for operational optimization:

- **Delivery Time Analysis**: Distribution and pattern identification for delivery timeframes
- **Cost Efficiency Analysis**: Cost structure examination by category and order size
- **Order Fulfillment Analysis**: Completion ratio visualization and performance metrics
- **Supplier Reliability Scoring**: Composite assessment of supplier performance

## 📂 Project Structure

- `app/`: Contains the Streamlit application code.
- `assets/`: Contains images and videos used in the Streamlit app.
- `data/`: Contains the data files used for analysis.
- `images/`: Screenshots of the Power BI dashboard.
- `notebooks/`: Jupyter notebooks with data visualizations and analysis.
- `powerbi/`: Contains the Power BI dashboard file.
- `summary_report/`: Contains the summary report and visualizations.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `LICENSE`: The license for the project.
- `README.md`: The main readme file for the project.
- `requirements.txt`: Lists the Python dependencies required for the project.

## 🌐 Interactive Web Application

The project now features a fully functional Streamlit web application with multiple analysis sections:

### Dashboard Section
- Interactive KPI cards with trend indicators
- Detailed metric explanations with visual guides
- Multi-dimensional visualization of order statuses
- Regional distribution analysis of suppliers and orders

### Supplier Analysis
- Comprehensive supplier performance evaluation
- Historical order tracking and pattern identification
- Time series analysis of supplier metrics
- Category-specific performance breakdown by supplier

### Product Analysis
- Product category performance comparison
- Cost structure visualization by product group
- Delivery reliability assessment by product type
- Order volume analysis across categories

### Correlation Analysis
- Interactive correlation explorer for key metrics
- Relationship strength visualization between variables
- Temporal correlation analysis for trend identification
- Category-based correlation patterns

### Prediction Section
- ML-powered OTIF prediction tool
- Scenario testing capabilities for order planning
- Key factor identification for OTIF optimization
- Historical comparison with similar orders

## 🔍 Enhanced Visualizations

The project now includes more sophisticated data visualizations:

- **KPI Explanation Cards**: Visual guides explaining metric calculations and business relevance
- **Distribution Charts**: Statistical visualization of key metrics distribution
- **Time Series Analysis**: Temporal patterns and trend visualization
- **Categorical Comparisons**: Side-by-side analysis of categories and suppliers
- **Cost Structure Charts**: Financial visualization of pricing and cost efficiency
- **Prediction Results**: Clear visual presentation of ML model outputs

## 🔄 Modern UI/UX Design

The user interface has been completely redesigned:

- **Responsive Layout**: Optimized for different screen sizes
- **Intuitive Navigation**: Logical flow between analysis sections
- **Visual Consistency**: Unified color scheme and design language
- **Interactive Elements**: Dynamic components for user engagement
- **Explanatory Content**: Contextual information and guidance throughout
- **Accessibility Features**: Design considerations for broader usability

### Access the Web App
You can access the Streamlit web app using the following URL:
[Supply Chain Project Web App](https://supply-chain-project-jy9fx2ga95cbnwbjcu3nyg.streamlit.app/)

Explore the data and gain valuable insights into the supply chain performance through our interactive web app. 🚚

## 🛠️ Requirements

- 💻 Power BI Desktop
- 🐍 Python 3.x
- 🐼 Pandas
- 📊 Seaborn
- 📉 Matplotlib
- 🌐 Streamlit
- 🤖 Scikit-learn
- 🧠 Joblib
- 📏 SciPy


## 📧 Contact

For any questions, you can contact me at:
- 📧 Email: jotaduranbon@gmail.com
- 💬 LinkedIn: [My LinkedIn Profile](www.linkedin.com/in/juan-duran-bon)

## 💡 Suggestions and Contributions

We welcome and appreciate any suggestions or contributions to improve this project. Here are some ways you can contribute:

- 🐞 **Report Issues**: If you encounter any bugs or have any issues while using the project, please open an issue on GitHub. Provide as much detail as possible, including steps to reproduce the issue and any relevant screenshots.

- 🌟 **Feature Requests**: If you have ideas for new features or enhancements, feel free to submit a feature request. Describe the feature in detail and explain how it would benefit the project.

- 🔄 **Submit Pull Requests**: If you would like to contribute code, you can fork the repository and submit a pull request. Make sure to follow the project's coding standards and include detailed commit messages. Before submitting a pull request, ensure that your code passes all tests and does not introduce any new issues.

- 📚 **Documentation**: Improving documentation is always helpful. If you find any part of the documentation unclear or incomplete, you can contribute by updating the documentation. This includes adding examples, clarifying instructions, and correcting any errors.

- 🧪 **Testing**: Help us improve the quality of the project by writing and running tests. You can add unit tests, integration tests, and end-to-end tests to ensure the project works as expected.

- 🗣️ **Feedback**: Your feedback is valuable to us. Let us know what you think about the project, what you like, and what could be improved. Your insights can help shape the future direction of the project.

To get started with contributing, please follow these steps:
1. 🍴 Fork the repository on GitHub.
2. 🖥️ Clone your forked repository to your local machine.
3. 🌿 Create a new branch for your changes.
4. ✏️ Make your changes and commit them with clear and descriptive messages.
5. 📤 Push your changes to your forked repository.
6. 🔀 Open a pull request to the main repository.

We appreciate your contributions and look forward to collaborating with you to make this project better!

Feel free to reach out if you have any questions or need assistance with your contributions.

## 📜 License

This project is licensed under the [MIT License](LICENSE).

## 👋 Farewell

Thank you for taking the time to explore this project! We hope you find it insightful and useful for your supply chain analysis needs. If you have any questions, suggestions, or just want to say hi, feel free to reach out. Your feedback and contributions are always welcome.

Happy analyzing! 🚀📊✨

