from flask import Flask, request, jsonify
from mlxtend.frequent_patterns import association_rules, apriori
import pickle
import pandas as pd
import re

app = Flask(__name__)

@app.route("/load_model", methods=["POST"])
def load_model():
    # Validate file format
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    file = request.files['file']
    
    if not file.filename.endswith('.csv'):
        return jsonify({"status": "error", "message": "Invalid file format, only CSV is allowed"}), 400
    try:
        data = pd.read_csv(file)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error reading CSV file: {str(e)}"}), 400
     

    data.dropna(subset=['bp_number'], inplace=True)
    
    period = request.form.get('period')  
    try:
        data['bp_number'] = data['bp_number'].astype(int)
        grouped_data = group_by_period(data, period)
        categorize_data = categorize_produk(grouped_data)
        processed_data = count_pivot_encode(categorize_data)
        result = model_mba(processed_data)
       
    except ValueError as e:
        return jsonify({"status": "error", "message": f"bp_number conversion error: {str(e)}"}), 400

    response_data = {
        "status": "success",
        "message": f"Processed data for period '{period}'",
        "data": result.to_dict(orient="records")
    }
    return jsonify(response_data)

def group_by_period(data, period):
    data['period'] = pd.to_datetime(data['period'], format='%Y%m', errors='coerce')
    data = data.sort_values(['bp_number', 'period']).reset_index(drop=True)
    data['year'] = data['period'].dt.year
    
    if period == 'bulanan':
        data['month'] = data['period'].dt.month  
        data['period_label'] = 'B' + data['month'].astype(str) + data['year'].astype(str)
        data['id_transaction'] = data['bp_number'].astype(str) + '_' + data['period_label']
        return data.drop(columns=['month', 'period_label'])
    
    elif period == 'pertiga_bulan':
        data['quarter'] = (data['period'].dt.month - 1) // 3 + 1
        data['period_label'] = 'Q' + data['quarter'].astype(str) + data['year'].astype(str)
        data['id_transaction'] = data['bp_number'].astype(str) + '_' + data['period_label']
        return data.drop(columns=['quarter', 'period_label'])
    
    elif period == 'perenam_bulan':
        data['semester'] = (data['period'].dt.month - 1) // 6 + 1
        data['period_label'] = 'S' + data['semester'].astype(str) + data['year'].astype(str)
        data['id_transaction'] = data['bp_number'].astype(str) + '_' + data['period_label']
        return data.drop(columns=['semester', 'period_label'])
    
    elif period == 'pertahun':
        data['period_label'] = 'Y' + data['year'].astype(str)
        data['id_transaction'] = data['bp_number'].astype(str) + '_' + data['period_label']
        return data.drop(columns=['year', 'period_label'])
    
    else:
        raise ValueError("Invalid period choice. Use 'bulanan', 'pertiga_bulan', 'perenam_bulan', or 'pertahun'.")

def categorize_produk(data):
    if 'amount' not in data.columns or 'product' not in data.columns:
        data['new_product'] = data['product'] if 'product' in data.columns else None
        return data[['id_transaction', 'new_product']]
    
    nunique_amount = data.groupby('product')['amount'].nunique()

    # Create categories and labels
    data['kategori_qcut'] = pd.qcut(data['amount'], 3, labels=None)
    data['kategori_qcut_label'] = data['kategori_qcut'].apply(
        lambda interval: f"{int(interval.left):,}-{int(interval.right):,}" 
                         if interval.left.is_integer() and interval.right.is_integer() 
                         else f"{interval.left:,.3f}-{interval.right:,.3f}"
    )

    # Create new product name without underscore or number modifications
    data['new_product'] = data.apply(
        lambda row: f"{row['product']} {row['kategori_qcut_label']}" 
        if nunique_amount[row['product']] >= 40 else row['product'], 
        axis=1
    )
    
    return data[['id_transaction', 'new_product']]
   
def count_pivot_encode(data) :
 
    item_count = data.groupby(['id_transaction', 'new_product'])['id_transaction'].count().reset_index(name="Count")
    item_count_pivot = item_count.pivot_table(index='id_transaction', columns='new_product', values='Count', aggfunc='sum').fillna(0)
    item_count_pivot = item_count_pivot.astype('int32')

    # Apply encoding
    item_count_pivot = item_count_pivot.applymap(lambda x: 1 if x >= 1 else 0)
    jumlah_baris, jumlah_kolom = item_count_pivot.shape
    print("Jumlah baris:", jumlah_baris)
    print("Jumlah kolom:", jumlah_kolom)
    #item_count_pivot.reset_index(inplace=True)

    return item_count_pivot
   
def model_mba(item_count_pivot) :
   frequent_itemsets = apriori(item_count_pivot, min_support=0.1, use_colnames=True)
   rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

   try:
       with open('model_mba.pkl', 'rb') as f:
           market_basket_model = pickle.load(f)
   except FileNotFoundError:
       market_basket_model = None

   market_basket_model = rules
   new_data = market_basket_model

   new_data['antecedents'] = new_data['antecedents'].apply(lambda x: list(x))
   new_data['consequents'] = new_data['consequents'].apply(lambda x: list(x))

   return new_data[['antecedents', 'consequents', 'support', 'confidence', 'lift']]



# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8089, debug=True)

# if __name__ == "__main__":
#     app.run(debug=True)
