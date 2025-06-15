from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os
import traceback # Untuk traceback error yang lebih detail

app = Flask(__name__)
CORS(app) # Izinkan semua origin untuk development

# --- DAPATKAN DIREKTORI TEMPAT app.py BERADA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"--- INFO DEBUGGING AWAL ---")
print(f"File app.py berada di: {BASE_DIR}")
print(f"Skrip Python dijalankan dari (Current Working Directory): {os.getcwd()}")
try:
    print(f"Isi dari direktori {BASE_DIR}:")
    for item in os.listdir(BASE_DIR):
        print(f"  - {item}")
except Exception as e:
    print(f"  Error saat membaca isi direktori {BASE_DIR}: {e}")
print(f"--- AKHIR INFO DEBUGGING AWAL ---")

# --- MUAT MODEL DAN SCALER ---
model_filename = 'model_rfV2.pkl'
scaler_filename = 'scaler.pkl'
encoder_filename = 'encoder.pkl' 

model_path = os.path.join(BASE_DIR, model_filename)
scaler_path = os.path.join(BASE_DIR, scaler_filename)
encoder_path = os.path.join(BASE_DIR, encoder_filename)

model = None
scaler = None
encoder_object_loaded = None

print(f"\nMencoba memuat model dari: {model_path}")
try:
    with open(model_path, 'rb') as f_model:
        model = pickle.load(f_model)
    print(f"Model '{model_filename}' berhasil dimuat.")
except FileNotFoundError:
    print(f"ERROR: File model '{model_filename}' TIDAK DITEMUKAN di '{model_path}'")
except Exception as e:
    print(f"ERROR saat memuat model '{model_filename}': {e}")
    traceback.print_exc()

print(f"\nMencoba memuat scaler dari: {scaler_path}")
try:
    with open(scaler_path, 'rb') as f_scaler:
        scaler = pickle.load(f_scaler)
    print(f"Scaler '{scaler_filename}' berhasil dimuat.")
    if hasattr(scaler, 'feature_names_in_'):
        print(f"[SCALER_INFO] Fitur yang dilihat scaler saat fit: {scaler.feature_names_in_}")
    if hasattr(scaler, 'n_features_in_'):
        print(f"[SCALER_INFO] Jumlah fitur yang dilihat scaler saat fit: {scaler.n_features_in_}")
except FileNotFoundError:
    print(f"ERROR: File scaler '{scaler_filename}' TIDAK DITEMUKAN di '{scaler_path}'")
except Exception as e:
    print(f"ERROR saat memuat scaler '{scaler_filename}': {e}")
    traceback.print_exc()

if os.path.exists(encoder_path):
    print(f"\nMencoba memuat encoder dari: {encoder_path} (untuk informasi saja, tidak untuk transform input fitur)")
    try:
        with open(encoder_path, 'rb') as f_encoder:
            encoder_object_loaded = pickle.load(f_encoder)
        print(f"Objek dari '{encoder_filename}' berhasil dimuat (tipe: {type(encoder_object_loaded).__name__}). Tidak akan digunakan untuk transform input fitur karena menggunakan mapping manual.")
    except Exception as e:
        print(f"ERROR saat memuat objek dari '{encoder_filename}': {e}")
        traceback.print_exc()
else:
    print(f"\nFile encoder '{encoder_filename}' tidak ditemukan. Mapping manual akan digunakan sepenuhnya.")

if not model or not scaler:
    print("\nPERINGATAN SERIUS: MODEL ATAU SCALER GAGAL DIMUAT! Prediksi tidak akan berfungsi.")
else:
    print("\nModel dan Scaler berhasil dimuat. Encoding fitur kategorikal input akan dilakukan dengan mapping manual.")

# --- KONFIGURASI FITUR ---
RAW_INPUT_COLUMNS = [ 
    'gender', 'married', 'dependents', 'education', 'selfEmployed',
    'applicantIncome', 'coapplicantIncome', 'loanAmount', 'loanTerm', # Frontend mengirim 'loanTerm'
    'creditHistory', 'propertyArea'
]

# Nama kolom SETELAH mapping manual & SEBELUM scaling.
# Ini harus SESUAI dengan nama kolom yang dilihat SCALER saat di-fit.
# Berdasarkan error, scaler mengharapkan 'Loan_Amount_Term'.
COLUMNS_AFTER_MAPPING_FOR_SCALER = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', # <--- PERUBAHAN DI SINI
    'Credit_History', 'Property_Area'
] 

# Fitur numerik yang akan di-scale (dengan nama kolom yang sudah disesuaikan untuk scaler)
NUMERICAL_FEATURES_FOR_SCALING = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'] # <--- PERUBAHAN DI SINI

# Urutan kolom FINAL yang diharapkan oleh MODEL setelah semua preprocessing
EXPECTED_COLUMNS_FOR_MODEL = COLUMNS_AFTER_MAPPING_FOR_SCALER # Asumsi sama dengan input scaler, sesuaikan jika perlu

# --- MAPPING DICTIONARIES ---
gender_map = {'Male': 1, 'Female': 0}
married_map = {'Yes': 1, 'No': 0}
dependents_map = {'0': 0, '1': 1, '2': 2, '3+': 3} 
education_map = {'Graduate': 0, 'Not Graduate': 1} 
self_employed_map = {'Yes': 1, 'No': 0}
property_area_map = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
DEFAULT_MAPPING_VALUE = -1 

@app.route('/predict_loan_approval', methods=['POST'])
def predict_loan_approval():
    print("\n--- Menerima Permintaan Prediksi (Mapping Manual & Penyesuaian Scaling) ---")
    if not model or not scaler:
        error_msg = "Model atau Scaler tidak termuat dengan benar di server. Prediksi dibatalkan."
        print(f"[PREDICT_ERROR] {error_msg}")
        return jsonify({'error': error_msg, 'prediction': None}), 500
    
    try:
        data_input_json = request.get_json(force=True)
        print(f"[PREDICT_INFO] Data mentah diterima dari frontend: {data_input_json}")

        input_df_raw = pd.DataFrame([data_input_json], columns=RAW_INPUT_COLUMNS)
        df_to_process = pd.DataFrame(index=[0])

        # 1. Mapping Manual dan Konversi Tipe
        df_to_process['ApplicantIncome'] = pd.to_numeric(input_df_raw['applicantIncome'], errors='coerce').fillna(0)
        df_to_process['CoapplicantIncome'] = pd.to_numeric(input_df_raw['coapplicantIncome'], errors='coerce').fillna(0)
        df_to_process['LoanAmount'] = pd.to_numeric(input_df_raw['loanAmount'], errors='coerce').fillna(0)
        # Ambil 'loanTerm' dari input mentah, tapi simpan sebagai 'Loan_Amount_Term' di df_to_process
        df_to_process['Loan_Amount_Term'] = pd.to_numeric(input_df_raw['loanTerm'], errors='coerce').fillna(0) # <--- PERUBAHAN DI SINI
        
        df_to_process['Gender'] = input_df_raw['gender'].map(gender_map).fillna(DEFAULT_MAPPING_VALUE)
        df_to_process['Married'] = input_df_raw['married'].map(married_map).fillna(DEFAULT_MAPPING_VALUE)
        df_to_process['Dependents'] = input_df_raw['dependents'].map(dependents_map).fillna(DEFAULT_MAPPING_VALUE)
        df_to_process['Education'] = input_df_raw['education'].map(education_map).fillna(DEFAULT_MAPPING_VALUE)
        df_to_process['Self_Employed'] = input_df_raw['selfEmployed'].map(self_employed_map).fillna(DEFAULT_MAPPING_VALUE)
        df_to_process['Property_Area'] = input_df_raw['propertyArea'].map(property_area_map).fillna(DEFAULT_MAPPING_VALUE)
        try:
            df_to_process['Credit_History'] = input_df_raw['creditHistory'].astype(float).astype(int)
        except ValueError:
            df_to_process['Credit_History'] = DEFAULT_MAPPING_VALUE
        
        print(f"[PREDICT_INFO] DataFrame setelah mapping manual (nama kolom disesuaikan untuk scaler):\n{df_to_process.head().to_string()}")

        for col in COLUMNS_AFTER_MAPPING_FOR_SCALER:
            if col not in df_to_process.columns:
                print(f"[PREDICT_WARNING] Kolom '{col}' yang diharapkan SCALER tidak ada setelah mapping, diisi {DEFAULT_MAPPING_VALUE}.")
                df_to_process[col] = DEFAULT_MAPPING_VALUE
        
        try:
            df_for_scaling_input = df_to_process[COLUMNS_AFTER_MAPPING_FOR_SCALER]
        except KeyError as e:
            print(f"[PREDICT_ERROR] KeyError saat menyusun kolom untuk scaler: {e}. Kolom yang ada: {df_to_process.columns.tolist()}")
            return jsonify({'error': f'Kesalahan penyusunan data untuk scaler: {e}', 'prediction': None}), 400
            
        print(f"[PREDICT_INFO] DataFrame yang akan diberikan ke scaler.transform():\n{df_for_scaling_input.to_string()}")
        print(f"[PREDICT_INFO] Kolom df_for_scaling_input: {df_for_scaling_input.columns.tolist()}")

        # 2. Scaling
        df_scaled_output = df_for_scaling_input.copy() # Mulai dengan DataFrame yang urutan kolomnya sudah benar
        
        print(f"[PREDICT_INFO] Menerapkan scaler...")
        try:
            # Scaler akan di-apply ke seluruh df_for_scaling_input.
            # Scaler yang benar (misal dari ColumnTransformer atau StandardScaler yang di-fit pada DataFrame
            # dengan tipe campuran) akan tahu kolom mana yang numerik untuk di-scale.
            scaled_values = scaler.transform(df_for_scaling_input)
            
            # Hasil transform adalah numpy array, ubah kembali ke DataFrame dengan nama kolom yang sama
            df_scaled_output = pd.DataFrame(scaled_values, columns=df_for_scaling_input.columns, index=df_for_scaling_input.index)
            print(f"[PREDICT_INFO] DataFrame setelah scaling:\n{df_scaled_output.head().to_string()}")

        except ValueError as ve:
             print(f"[PREDICT_ERROR] ValueError saat scaling: {ve}")
             print(f"[PREDICT_HINT] Pastikan DataFrame yang diberikan ke scaler.transform() ('df_for_scaling_input') memiliki SEMUA kolom (dengan nama dan urutan yang benar) yang dilihat scaler saat fit.")
             if hasattr(scaler, 'feature_names_in_'):
                 print(f"[SCALER_INFO_ERROR] Scaler mengharapkan fitur: {scaler.feature_names_in_}")
             print(f"[SCALER_INFO_ERROR] Fitur yang diberikan: {df_for_scaling_input.columns.tolist()}")
             traceback.print_exc()
             return jsonify({'error': f'Error saat scaling (ValueError): {ve}', 'prediction': None}), 400
        except Exception as e:
            print(f"[PREDICT_ERROR] Gagal saat scaling: {e}")
            traceback.print_exc()
            return jsonify({'error': f'Error saat scaling data: {e}', 'prediction': None}), 400

        # 3. Susun Ulang Kolom Sesuai Urutan yang Diharapkan Model (jika berbeda dari output scaler)
        print(f"[PREDICT_INFO] Kolom yang diharapkan model: {EXPECTED_COLUMNS_FOR_MODEL}")
        print(f"[PREDICT_INFO] Kolom yang ada di df_scaled_output sebelum reorder untuk model: {df_scaled_output.columns.tolist()}")
        try:
            # Jika EXPECTED_COLUMNS_FOR_MODEL sama dengan COLUMNS_AFTER_MAPPING_FOR_SCALER,
            # dan df_scaled_output sudah memiliki kolom tersebut, maka final_input_df akan sama.
            missing_cols_for_model = [col for col in EXPECTED_COLUMNS_FOR_MODEL if col not in df_scaled_output.columns]
            if missing_cols_for_model:
                print(f"[PREDICT_ERROR] Kolom berikut hilang dan tidak bisa diberikan ke model: {missing_cols_for_model}")
                raise ValueError(f"Kolom yang dibutuhkan model hilang: {missing_cols_for_model}")

            final_input_df = df_scaled_output[EXPECTED_COLUMNS_FOR_MODEL]
            print(f"[PREDICT_INFO] DataFrame final untuk model (sebelum .values):\n{final_input_df.to_string()}")
        except KeyError as ke:
            print(f"[PREDICT_ERROR] KeyError saat menyusun ulang kolom untuk model: {ke}. Kolom yang ada: {df_scaled_output.columns.tolist()}.")
            traceback.print_exc()
            return jsonify({'error': f'Kesalahan internal: Kolom {ke} tidak ada setelah preprocessing akhir.', 'prediction': None}), 500
            
        model_input_array = final_input_df.values
        print(f"[PREDICT_INFO] Bentuk input array untuk model: {model_input_array.shape}")

        # --- LAKUKAN PREDIKSI ---
        prediksi = model.predict(model_input_array)
        prediksi_proba = None
        if hasattr(model, 'predict_proba'):
            prediksi_proba = model.predict_proba(model_input_array)
            print(f"[PREDICT_INFO] Probabilitas prediksi (mentah): {prediksi_proba}")
        
        hasil_prediksi = int(prediksi[0]) 
        print(f"[PREDICT_INFO] Hasil prediksi akhir (0 atau 1): {hasil_prediksi}")

        return jsonify({
            'prediction': hasil_prediksi,
            'probabilities': prediksi_proba[0].tolist() if prediksi_proba is not None and len(prediksi_proba) > 0 else None
        })

    except Exception as e:
        print(f"[PREDICT_CRITICAL_ERROR] Exception tidak terduga di endpoint prediksi: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Terjadi kesalahan internal server yang tidak terduga: {str(e)}', 'prediction': None}), 500

if __name__ == '__main__':
    print("\nMenjalankan server Flask...")
    app.run(host='0.0.0.0', port=5000, debug=True)