# 📊✨ Summary Report ✨📊

This report provides a detailed analysis of the purchase orders and deliveries dataset. Below are the generated charts, the new calculated columns, and their added value to the analysis.

## 📈 Generated Charts 📈

### 1. 📉 Order Status Distribution
- **Description**: This bar plot shows the distribution of order statuses.
- **Value**: Helps to understand the frequency of different order statuses, identifying patterns and possible issues.

### 2. 🌍 Origin Country Distribution
- **Description**: This bar plot shows the distribution of origin countries.
- **Value**: Allows analyzing the diversity of suppliers and identifying the most common origin countries.

### 3. 💲 Unit Price by Category
- **Description**: This box plot shows the distribution of unit prices by category.
- **Value**: Helps to compare unit prices across different categories, identifying significant differences.

### 4. 🏷️ Unit Price by Subcategory
- **Description**: This box plot shows the distribution of unit prices by subcategory.
- **Value**: Allows comparing unit prices across different subcategories, identifying significant differences.

### 5. 💵 Cost vs Unit Price
- **Description**: This scatter plot shows the relationship between product cost and unit price.
- **Value**: Helps to evaluate the pricing strategy and identify any discrepancies between cost and unit price.

### 6. ⏳ Delivery Days vs Reception Days
- **Description**: This scatter plot shows the relationship between delivery days and reception days.
- **Value**: Allows analyzing the efficiency of the delivery and reception process, identifying any delays.

### 7. 📅 Total Amount Over Time
- **Description**: This line plot shows how the total amount varies over time.
- **Value**: Helps to identify trends and seasonal patterns in total amounts.

### 8. 📊 Total Amount by Category
- **Description**: This bar plot shows the total amount by category.
- **Value**: Allows analyzing the contribution of each category to the total amount, identifying the most profitable categories.

### 9. ⏰ On-time Deliveries by Provider
- **Description**: This bar plot shows the number of on-time deliveries by provider.
- **Value**: Helps to evaluate the performance of different providers in terms of delivery punctuality.

### 10. ✅ OTIF Deliveries by Provider
- **Description**: This bar plot shows the number of OTIF (On Time In Full) deliveries by provider.
- **Value**: Allows evaluating the performance of different providers in terms of complete and on-time deliveries.

### 11. 📦 Histogram of Ordered Product Quantity
- **Description**: This histogram shows the distribution of ordered product quantities.
- **Value**: Helps to understand the frequency of different ordered quantities, identifying patterns and possible outliers.

### 12. 💲 Histogram of Unit Price
- **Description**: This histogram shows the distribution of product unit prices.
- **Value**: Allows analyzing the variability of unit prices and detecting outliers.

### 13. 📅 Order Amount Over Time
- **Description**: This line chart shows how the order amount varies over time.
- **Value**: Helps to identify trends and seasonal patterns in order amounts.

### 14. 📦 Unit Price by Order Status
- **Description**: This box plot shows the distribution of unit prices by order status.
- **Value**: Allows comparing unit prices across different order statuses, identifying possible significant differences.

### 15. 🔄 Ordered Quantity vs Received Quantity
- **Description**: This scatter plot shows the relationship between ordered quantity and received quantity.
- **Value**: Helps to evaluate the accuracy of deliveries in terms of quantity, identifying discrepancies between ordered and received quantities.

### 16. ⏳ Distribution of Delivery Days (<= 30)
- **Description**: This histogram shows the distribution of delivery days, limited to a maximum of 30 days.
- **Value**: Allows analyzing the efficiency of deliveries, identifying possible delays and their frequency.

### 17. ⏰ On-time Deliveries
- **Description**: This bar chart shows the number of on-time deliveries.
- **Value**: Helps to evaluate performance in terms of delivery punctuality.

### 18. ✅ OTIF Deliveries
- **Description**: This bar chart shows the number of OTIF (On Time In Full) deliveries.
- **Value**: Allows evaluating performance in terms of complete and on-time deliveries.

## 🧮 New Calculated Columns 🧮

### 1. ⏱️ `ontime`
- **Description**: Indicates whether the delivery was on time (`True`) or not (`False`).
- **Calculation**: `data['ontime'] = data['fecha_entrega'] >= data['fecha_recibido']`
- **Value**: Allows evaluating the punctuality of deliveries.

### 2. 📦 `OTIF`
- **Description**: Indicates whether the delivery was on time and in full (`True`) or not (`False`).
- **Calculation**: `data['OTIF'] = (data['ontime']) & (data['cant_prod_odc'] == data['cant_recibida'])`
- **Value**: Allows evaluating performance in terms of complete and on-time deliveries.

### 3. 📅 `delivery_days`
- **Description**: Number of days between the order date and the delivery date.
- **Calculation**: `data['delivery_days'] = (data['fecha_entrega'] - data['fecha_odc']).dt.days`
- **Value**: Allows analyzing the efficiency of deliveries.

### 4. 📅 `reception_days`
- **Description**: Number of days between the order date and the reception date.
- **Calculation**: `data['reception_days'] = (data['fecha_recibido'] - data['fecha_odc']).dt.days`
- **Value**: Allows analyzing the efficiency of the reception process.

### 5. 📊 `percentage_received`
- **Description**: Percentage of the received quantity relative to the ordered quantity.
- **Calculation**: `data['percentage_received'] = (data['cant_recibida'] / data['cant_prod_odc']) * 100`
- **Value**: Allows evaluating the accuracy of deliveries in terms of quantity.

### 6. 💰 `amount_difference`
- **Description**: Difference between the ordered amount and the received amount.
- **Calculation**: `data['amount_difference'] = data['monto_odc'] - data['monto_recibido']`
- **Value**: Allows identifying monetary discrepancies between ordered and received amounts.

### 7. 💵 `total_amount`
- **Description**: Total amount of the order (ordered quantity * unit price).
- **Calculation**: `data['total_amount'] = data['cant_prod_odc'] * data['prec_unt']`
- **Value**: Provides a measure of the total value of each order.

## 📋 Conclusion 📋

This analysis provides a comprehensive view of the performance of purchase orders and deliveries, allowing the identification of patterns, trends, and potential areas for improvement. The new calculated columns add value to the dataset by providing additional metrics that facilitate a more detailed and accurate evaluation.

We hope this report is useful to you! 📊✨