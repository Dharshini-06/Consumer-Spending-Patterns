# Consumer-Spending-Patterns

An interactive data analytics dashboard built using Python and Streamlit to analyze consumer spending behavior, visualize trends, and perform customer segmentation using K-Means clustering.

This project enables users to upload transaction datasets and gain meaningful business insights through real-time analytics, machine learning, and interactive visualizations.

## Project Description

The Consumer Spending Patterns dashboard focuses on understanding customer purchase behavior. It processes transaction data, cleans it automatically, and presents insights such as total revenue, spending trends, customer segmentation, and anomaly detection. The application is designed with a modern UI and interactive controls to enhance data exploration.

## Features

- Upload CSV or Excel datasets
- Automatic data cleaning (duplicate and null removal)
- Interactive KPI metrics
- Category-wise spending analysis
- Monthly revenue trend visualization
- Gender-based spending distribution
- Customer segmentation using K-Means clustering
- PCA-based cluster visualization
- Outlier detection using distance metrics
- Interactive sidebar filters and sliders

## Machine Learning Techniques

- K-Means Clustering for customer segmentation
- StandardScaler for feature normalization
- Principal Component Analysis (PCA) for dimensionality reduction
- Distance-based outlier detection

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Matplotlib
- Scikit-learn

## Project Structure

Consumer-Spending-Patterns/
├── dashboard.py
├── requirements.txt
├── README.md
└── dataset.csv

## Dataset Requirements

The dataset may contain the following columns:
- Customer ID
- Transaction Amount
- Category
- Gender
- Date

The dashboard automatically adapts based on the available columns in the dataset.

## How to Run the Project

Clone the repository:
git clone https://github.com/your-username/Consumer-Spending-Patterns.git

Navigate to the project directory:
cd Consumer-Spending-Patterns

Install the required dependencies:
pip install -r requirements.txt

Run the Streamlit application:
streamlit run dashboard.py

## Output

- Interactive consumer spending dashboard
- Visual insights and analytics
- Customer segmentation results
- Outlier identification
- Business intelligence reports

## Project Demo 

https://github.com/user-attachments/assets/3b65ad45-cf0d-4a45-99cd-ead6d3e55053

## Use Cases

- Retail and e-commerce analytics
- Customer behavior analysis
- Business intelligence dashboards
- Academic mini projects
- Machine learning practice

## Author

Dharshini G | BTECH Information Technology
