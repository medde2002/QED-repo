from flask import Flask, jsonify, request, send_file
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import io
import os
from datetime import datetime
import traceback

# Initialize app
app = Flask(__name__, static_folder='static')

# Global variables to store data during session
par_curve_data = None
spot_curve_data = None
continuous_curve_data = None

def year_to_file(year):
    
    # Map a given year to its corresponding CSV filename based on my file naming convention.
    # Example: 2025 -> daily-treasury-rates (21).csv
    
    file_num = 2046 - year
    return f"static/data/daily-treasury-rates ({file_num}).csv"

def load_treasury_data(year):
    
    # Load Treasury data for a given year.
    
    file_path = year_to_file(year)
    if not os.path.exists(file_path):
        raise Exception(f"Treasury data for {year} not found.")
    
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def find_closest_row(df, date_str):
    
    # Find the closest available date in the Treasury data to the user-selected date.
    
    target = pd.to_datetime(date_str)
    df_sorted = df.sort_values('Date')
    idx = df_sorted['Date'].searchsorted(target)
    if idx == len(df_sorted):
        idx -= 1
    return df_sorted.iloc[idx]

def calculate_par_curve(row):
    
    # This is to accomodate the different file formats
    # For instance, while debugging, I noticed that the format for 2009 - Onwards is different 

    possible_columns = {
        '1 Mo': ['1 Mo', '1M'],
        '2 Mo': ['2 Mo', '2M'],
        '3 Mo': ['3 Mo', '3M'],
        '6 Mo': ['6 Mo', '6M'],
        '1 Yr': ['1 Yr', '1 YR', '1Y'],
        '2 Yr': ['2 Yr', '2 YR', '2Y'],
        '3 Yr': ['3 Yr', '3 YR', '3Y'],
        '5 Yr': ['5 Yr', '5 YR', '5Y'],
        '7 Yr': ['7 Yr', '7 YR', '7Y'],
        '10 Yr': ['10 Yr', '10 YR', '10Y'],
        '20 Yr': ['20 Yr', '20 YR', '20Y'],
        '30 Yr': ['30 Yr', '30 YR', '30Y']
    }
    
    labels = []
    tenor_years = []
    yields = []
    
    for label, alternatives in possible_columns.items():
        for alt in alternatives:
            if alt in row and not pd.isna(row[alt]):
                labels.append(label)
                yields.append(row[alt])
                # Convert label to maturity in years
                if 'Mo' in label:
                    tenor_years.append(int(label.split(' ')[0]) / 12)
                else:
                    tenor_years.append(int(label.split(' ')[0]))
                break  # stop once a valid alternative is found

    return {
        'date': row['Date'].strftime('%Y-%m-%d'),
        'labels': labels,
        'tenors': tenor_years,
        'yields': yields
    }

def bootstrap_spot_rates(par_curve, face_value=100, coupon_freq=2):
    
    # Bootstrap spot rates from par yields.
    
    tenors = np.array(par_curve['tenors'])
    par_yields = np.array(par_curve['yields'])
    spot_rates = []

    for i, (tenor, par_yield) in enumerate(zip(tenors, par_yields)):
        annual_coupon_rate = par_yield / 100
        coupon_per_period = (annual_coupon_rate * face_value) / coupon_freq
        periods = int(tenor * coupon_freq)

        if tenor <= 1:
            spot_rates.append(annual_coupon_rate)
            continue

        price = face_value
        pv_coupons = 0

        for t in range(1, periods):
            year_t = t / coupon_freq
            idx = np.searchsorted(tenors[:i], year_t, side='right') - 1
            idx = max(idx, 0)

            if idx < len(spot_rates):
                prev_spot = spot_rates[idx]

                if idx + 1 < len(spot_rates) and idx + 1 < i and tenors[idx + 1] <= year_t:
                    next_spot = spot_rates[idx + 1]
                    next_tenor = tenors[idx + 1]
                    prev_tenor = tenors[idx]
                    discount_rate = prev_spot + (next_spot - prev_spot) * (year_t - prev_tenor) / (next_tenor - prev_tenor)
                else:
                    discount_rate = prev_spot

                pv_coupons += coupon_per_period / ((1 + discount_rate / coupon_freq) ** t)

        final_payment = face_value + coupon_per_period
        spot_rate = coupon_freq * ((final_payment / (price - pv_coupons)) ** (1 / periods) - 1)
        spot_rates.append(spot_rate)

    return {
        'tenors': tenors.tolist(),
        'par_yields': par_yields.tolist(),
        'spot_rates': [round(r * 100, 4) for r in spot_rates]
    }

def calculate_continuous_curve(spot_data):
    
   # Generate a continuous monthly zero-rate curve using cubic spline interpolation.
    
    tenors = np.array(spot_data['tenors'])
    spot_rates = np.array(spot_data['spot_rates'])

    monthly_tenors = np.linspace(1/12, 30, 360)
    cs = CubicSpline(tenors, spot_rates)
    monthly_spot_rates = cs(monthly_tenors)

    labels = [f"{int(tenor)}Y" if i % 12 == 0 else "" for i, tenor in enumerate(monthly_tenors)]

    return {
        'monthly_tenors': monthly_tenors.tolist(),
        'monthly_spot_rates': [round(r, 4) for r in monthly_spot_rates],
        'labels': labels
    }

#######################
# Flask API Endpoints #
#######################

@app.route('/')
def index():
    return app.send_static_file('index.html')

# API Endpoint to generate the par yield curve for a given date

@app.route('/api/par-curve', methods=['GET'])
def get_par_curve():
    global par_curve_data
    try:
        date_str = request.args.get('date')
        pd.to_datetime(date_str)  # validate date
        df = load_treasury_data(pd.to_datetime(date_str).year)
        row = find_closest_row(df, date_str)
        par_curve_data = calculate_par_curve(row)

        return jsonify({
            'status': 'success',
            'data': {
                'date': par_curve_data['date'],
                'labels': par_curve_data['labels'],
                'yields': par_curve_data['yields'],
                'tenors': par_curve_data['tenors']
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

# API endpoint to bootstrap spot rates from the par curve.
# Requires par curve to be geenrated first.

@app.route('/api/spot-curve', methods=['GET'])
def get_spot_curve():
    global spot_curve_data, par_curve_data
    try:
        if not par_curve_data:
            return jsonify({'status': 'error', 'message': 'Par curve not generated yet.'}), 400

        face_value = float(request.args.get('faceValue', 100))
        spot_curve_data = bootstrap_spot_rates(par_curve_data, face_value)

        return jsonify({
            'status': 'success',
            'data': {
                'tenors': spot_curve_data['tenors'],
                'spotRates': spot_curve_data['spot_rates']
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

# API endpoint to generate continuous monthly zero curve.
# Requires spot curve to be generated first

@app.route('/api/continuous-curve', methods=['GET'])
def get_continuous_curve():
    global continuous_curve_data, spot_curve_data
    try:
        if not spot_curve_data:
            return jsonify({'status': 'error', 'message': 'Spot curve not generated yet.'}), 400

        continuous_curve_data = calculate_continuous_curve(spot_curve_data)

        return jsonify({
            'status': 'success',
            'data': {
                'monthly_tenors': continuous_curve_data['monthly_tenors'],
                'monthly_spot_rates': continuous_curve_data['monthly_spot_rates'],
                'labels': continuous_curve_data['labels']
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

# API endpoint to export all curve data to Excel
# We need to generate all the curves first
@app.route('/api/export-data', methods=['GET'])
def export_data():
    global par_curve_data, spot_curve_data, continuous_curve_data
    try:
        if not par_curve_data or not spot_curve_data or not continuous_curve_data:
            return jsonify({'status': 'error', 'message': 'Data incomplete. Generate all curves first.'}), 400

        # Create DataFrames for Excel
        par_df = pd.DataFrame({
            'Maturity (Years)': par_curve_data['tenors'],
            'Par Yield (%)': par_curve_data['yields'],
            'Spot Rate (%)': spot_curve_data['spot_rates']
        })

        monthly_df = pd.DataFrame({
            'Maturity (Years)': continuous_curve_data['monthly_tenors'],
            'Monthly Spot Rate (%)': continuous_curve_data['monthly_spot_rates']
        })

        # Create Excel file in memory
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            par_df.to_excel(writer, sheet_name='Par and Spot Curve', index=False)
            monthly_df.to_excel(writer, sheet_name='Monthly Zero Curve', index=False)
        buffer.seek(0)

        # Send file as attachment
        return send_file(
            buffer,
            as_attachment=True,
            download_name='yield_curve_data.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Run app locally
if __name__ == '__main__':
    print("Server starting at http://localhost:8080")
    app.run(debug=True, port=8080)
