import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the Excel file
data = pd.read_excel('PharmaData.xlsx')

# Visualization 1: Total Sales by Branch Name and Department
sales_by_branch = data.groupby(['Branch Name', 'Department'])['Sales'].sum().reset_index()

# Visualization 1: Bar chart
plt.figure(figsize=(10, 6))
plt.bar(sales_by_branch['Branch Name'], sales_by_branch['Sales'], color='b')
plt.xlabel('Branch Name')
plt.ylabel('Total Sales')
plt.title('Total Sales by Branch and Department')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualization 2: Average Sales per Quantity by Product Category
data['Avg_Sales_Per_Qty'] = data['Sales'] / data['Qty']
avg_sales_by_category = data.groupby('Product Category')['Avg_Sales_Per_Qty'].mean().reset_index()

# Visualization 2: Horizontal bar chart
plt.figure(figsize=(10, 20))
plt.barh(avg_sales_by_category['Product Category'], avg_sales_by_category['Avg_Sales_Per_Qty'], color='g')
plt.xlabel('Average Sales per Quantity')
plt.ylabel('Product Category')
plt.title('Average Sales per Quantity by Product Category')
plt.tight_layout()
plt.show()
