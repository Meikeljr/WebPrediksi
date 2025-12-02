from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import pandas as pd
import statsmodels.api as sm
import os
import pickle
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'kunci_rahasia_dan_unik_anda' 

# =======================================================
# FAKE USER DATABASE
# =======================================================
USERS = {
    "admin": {"password": "admin123", "id": 1, "name": "Admin Penjualan"}
}

# =======================================================
# USER CLASS UNTUK FLASK-LOGIN
# =======================================================
class User(UserMixin):
    def __init__(self, user_id, username, name):
        self.id = user_id
        self.username = username
        self.name = name

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    for username, user_data in USERS.items():
        if str(user_data['id']) == user_id:
            return User(user_data['id'], username, user_data['name'])
    return None

# =======================================================
# KONFIGURASI DATA STATIS DAN VARIABEL MODEL
# =======================================================
STATIC_DATA_PATH = 'data/penjualan.csv'
DETAIL_DATA_PATH = 'data/penjualandetail.csv'
DELIMITER = ',' 

# PERIODE NUMERICAL MAPPING (idul_adha = 0 (Base/Reference), idul_fitri = 1)
PERIODE_NUMERIC_MAPPING = {
    "idul_fitri": 1,
    "idul_adha": 0
}
PERIODE_VAR = "Periode"

# UKURAN NUMERICAL MAPPING (lainnya = 0 (Base/Reference), sedang = 1)
UKURAN_NUMERIC_MAPPING = {
    "sedang": 1,
    "lainnya": 0 
}
UKURAN_VAR = "Ukuran"

CATEGORICAL_MAPPING = {
    # 'Periode' & 'Ukuran' Dikeluarkan dari sini karena di-encode secara numerik 0/1
    "Jenis Kue": ["Beng_beng", "kue_lainnya", "lidah_kucing", "nastar", "putri_salju_vanilla", "rambutan", "sagu_keju", "semprit_susu"],
}
NUMERIC_VAR = "Tahun"

REFERENCE_CATEGORIES = {
    # 'Periode' & 'Ukuran' Dikeluarkan dari sini
    "Jenis Kue": "kue_lainnya", 
}

# MAPPING UNTUK PEMBENTUKAN FORM PADA prediksi.html 
FORM_CATEGORICAL_MAPPING = {
    PERIODE_VAR: list(PERIODE_NUMERIC_MAPPING.keys()),
    UKURAN_VAR: list(UKURAN_NUMERIC_MAPPING.keys()),
    **CATEGORICAL_MAPPING
}

# =======================================================
# UTILITY FUNCTIONS
# =======================================================

def encode_model(model):
    return base64.b64encode(pickle.dumps(model)).decode('utf-8')

def decode_model(model_str):
    return pickle.loads(base64.b64decode(model_str.encode('utf-8')))

def drop_reference_category(data):
    # 1. Handle 'Periode' using numerical encoding (0 or 1)
    if PERIODE_VAR in data.columns:
        # Mengganti nilai string dengan nilai numerik sesuai mapping
        data[PERIODE_VAR] = data[PERIODE_VAR].map(PERIODE_NUMERIC_MAPPING)
        # Hapus baris jika mapping gagal
        data.dropna(subset=[PERIODE_VAR], inplace=True) 
    
    # 2. Handle 'Ukuran' using numerical encoding (0 or 1)
    if UKURAN_VAR in data.columns:
        data[UKURAN_VAR] = data[UKURAN_VAR].map(UKURAN_NUMERIC_MAPPING)
        data.dropna(subset=[UKURAN_VAR], inplace=True) 

    # 3. Handle other categorical variables using dummy encoding (one-hot)
    # Filter kolom kategorikal yang tersisa untuk dummify (yaitu selain 'Periode' dan 'Ukuran')
    cols_to_dummify = [col for col in REFERENCE_CATEGORIES.keys() if col in data.columns]
    
    # Buat variabel dummy untuk kolom-kolom yang tersisa
    data_dummies = pd.get_dummies(data, columns=cols_to_dummify, drop_first=False)
    
    # Hapus kolom referensi untuk variabel dummy
    cols_to_drop = []
    for category, ref_option in REFERENCE_CATEGORIES.items():
        if category in cols_to_dummify:
            col_to_drop = f"{category}_{ref_option}"
            if col_to_drop in data_dummies.columns:
                cols_to_drop.append(col_to_drop)
            
    data_final = data_dummies.drop(columns=cols_to_drop, errors='ignore')
    return data_final

def build_regression_model(data_path):
    data_raw = pd.read_csv(data_path, sep=DELIMITER, header=0)
    
    Y_NAME = data_raw.columns[0].strip()
    
    for col in [Y_NAME, NUMERIC_VAR]:
        if col in data_raw.columns:
             data_raw[col] = pd.to_numeric(data_raw[col], errors='coerce')
        
    data_raw.dropna(inplace=True)
    
    data_model = drop_reference_category(data_raw) 
    
    Y = data_model[Y_NAME]
    X = data_model.drop(columns=[Y_NAME])
    
    X = X.astype(float)
    Y = Y.astype(float)
    
    X = X.loc[:, (X != X.iloc[0]).any()] 
    
    if X.empty: raise ValueError("Model tidak dapat dibangun karena tidak ada variabel independen.")
    
    X = sm.add_constant(X, has_constant='add')
    model = sm.OLS(Y, X).fit()
    
    session['fitted_model'] = encode_model(model)
    session['y_name'] = Y_NAME
    session['trained_features'] = X.columns.tolist()
    
    return model, Y_NAME, X.columns.tolist()

def calculate_prediction(model, trained_features, Y_NAME, form_data):
    input_dict = {'const': 1}
    
    # 1. Handle NUMERIC_VAR ('Tahun')
    tahun_val = pd.to_numeric(form_data.get(NUMERIC_VAR), errors='coerce')
    if tahun_val is None or pd.isna(tahun_val):
         raise ValueError("Input 'Tahun' harus berupa angka.")
    input_dict[NUMERIC_VAR] = tahun_val
    
    # 2. Handle PERIODE_VAR ('Periode') - Numerically encoded (0 or 1)
    periode_selected = form_data.get(PERIODE_VAR)
    if not periode_selected or periode_selected not in PERIODE_NUMERIC_MAPPING:
        raise ValueError(f"Input '{PERIODE_VAR}' tidak valid.")
    input_dict[PERIODE_VAR] = PERIODE_NUMERIC_MAPPING[periode_selected]
    
    # 3. Handle UKURAN_VAR ('Ukuran') - Numerically encoded (0 or 1)
    ukuran_selected = form_data.get(UKURAN_VAR)
    if not ukuran_selected or ukuran_selected not in UKURAN_NUMERIC_MAPPING:
        raise ValueError(f"Input '{UKURAN_VAR}' tidak valid.")
    input_dict[UKURAN_VAR] = UKURAN_NUMERIC_MAPPING[ukuran_selected]
    
    # 4. Handle other CATEGORICAL_MAPPING variables (dummy encoded)
    all_dummy_cols = [f"{cat}_{opt}" for cat, options in CATEGORICAL_MAPPING.items() for opt in options]
    for col in all_dummy_cols:
        if col in trained_features:
            input_dict[col] = 0

    for category_name, options in CATEGORICAL_MAPPING.items():
        selected_option = form_data.get(category_name)
        
        selected_col_name = f"{category_name}_{selected_option}"
        
        # Cek apakah variabel ini merupakan variabel yang di-dummy (bukan variabel referensi)
        if selected_option and selected_option != REFERENCE_CATEGORIES.get(category_name):
            if selected_col_name in trained_features:
                 input_dict[selected_col_name] = 1
        
    input_data = pd.DataFrame([input_dict], index=[0])
    
    # Pastikan urutan kolom sesuai dengan model yang dilatih
    required_cols = [col for col in model.params.index if col in input_data.columns]
    
    input_data = input_data[required_cols].reindex(columns=model.params.index, fill_value=0)
    
    prediction = model.predict(input_data)[0]
    
    final_inputs = {}
    final_inputs[NUMERIC_VAR] = form_data.get(NUMERIC_VAR)
    
    # Masukkan Periode dan Ukuran ke dalam final_inputs
    final_inputs[PERIODE_VAR] = form_data.get(PERIODE_VAR)
    final_inputs[UKURAN_VAR] = form_data.get(UKURAN_VAR)
    
    for category in CATEGORICAL_MAPPING.keys():
        final_inputs[category] = form_data.get(category)
            
    return prediction, final_inputs

# =======================================================
# 1. ROUTES AUTENTIKASI (LOGIN & LOGOUT)
# =======================================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('beranda'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user_data = USERS.get(username)
        
        if user_data and user_data['password'] == password:
            user_obj = User(user_data['id'], username, user_data['name'])
            login_user(user_obj)
            flash('Login berhasil!', 'success')
            return redirect(url_for('beranda'))
        else:
            flash('Login gagal. Periksa username dan password Anda.', 'danger')

    return render_template('login.html')

@app.route('/logout')
@login_required 
def logout():
    logout_user()
    session.pop('fitted_model', None)
    session.pop('y_name', None)
    session.pop('trained_features', None)
    flash('Anda telah berhasil logout.', 'success')
    return redirect(url_for('login'))

# =======================================================
# 2. ROUTES APLIKASI UTAMA
# =======================================================

@app.before_request
def check_model():
    if current_user.is_authenticated and 'fitted_model' not in session and (request.path == url_for('prediksi') or request.path == url_for('ringkasan_model')):
        try:
            build_regression_model(STATIC_DATA_PATH)
        except Exception as e:
            flash(f"Model gagal dibangun saat startup: {e}", 'danger')
            return redirect(url_for('beranda'))


@app.route('/')
@app.route('/beranda')
@login_required
def beranda():
    return render_template('beranda.html')

@app.route('/data_penjualan')
@login_required
def data_penjualan():
    try:
        data = pd.read_csv(DETAIL_DATA_PATH, sep=DELIMITER, header=0)
        data.columns = [col.strip().replace(' ', '_') for col in data.columns]
        data_html = data.to_html(index=False, classes='table table-striped table-bordered display', table_id='penjualanTable')
        
        return render_template('data_penjualan.html', data_html=data_html)
    except Exception as e:
        flash(f"Gagal memuat data penjualan: {e}", 'danger')
        return redirect(url_for('beranda'))
    

# =======================================================
# 3. ROUTE RINGKASAN MODEL BARU
# =======================================================

@app.route('/ringkasan_model')
@login_required
def ringkasan_model():
    if 'fitted_model' not in session:
        flash('Model belum dibangun. Silakan coba buka halaman Prediksi terlebih dahulu.', 'warning')
        return redirect(url_for('prediksi'))
        
    try:
        model = decode_model(session['fitted_model'])
        y_name = session['y_name']
        
        metrics = {
            'R-squared': f"{model.rsquared:.4f}",
            'Adj. R-squared': f"{model.rsquared_adj:.4f}",
            'F-statistic': f"{model.fvalue:.4f}",
            'Prob (F-statistic)': f"{model.f_pvalue:.4f}",
            'N Observasi': int(model.nobs), 
            'Jml Variabel Independen': len(model.params) - 1
        }
        
        equation = f"{y_name} = {model.params['const']:.4f}"
        for col in model.params.index:
            if col != 'const':
                equation += f" + {model.params[col]:.4f} * {col.replace('_', ' ')}"

        if len(model.summary().tables) > 1:
            summary_table_html = model.summary().tables[1].as_html()
            summary_table = summary_table_html.replace('<th>', '<th scope="col">').replace('<table', '<table class="table table-sm table-bordered table-striped"').replace('style="', 'data-style="')

        else:
            summary_table = "<p>Tabel Koefisien tidak tersedia.</p>"
            
    except Exception as e:
        flash(f"Gagal memuat atau memproses ringkasan model: {e}", 'danger')
        return redirect(url_for('beranda'))

    return render_template('ringkasan_model.html', 
                           y_name=y_name,
                           metrics=metrics,
                           equation=equation,
                           summary_table=summary_table,
                           trained_features=session['trained_features'])


# =======================================================
# 4. ROUTE PREDIKSI
# =======================================================

@app.route('/prediksi', methods=['GET', 'POST'])
@login_required
def prediksi():
    prediction_result = None
    prediction_inputs = None
    error = None
    
    if 'fitted_model' not in session:
        try:
             build_regression_model(STATIC_DATA_PATH)
        except Exception as e:
            flash(f"Model gagal dibangun saat mencoba prediksi: {e}", 'danger')
            return redirect(url_for('beranda'))
        
    try:
        model = decode_model(session['fitted_model'])
        y_name = session['y_name']
        trained_features = session['trained_features']
        
        metrics = {
            'R-squared': f"{model.rsquared:.4f}",
            'Adj. R-squared': f"{model.rsquared_adj:.4f}",
        }
        
        if request.method == 'POST':
            prediction_result, prediction_inputs = calculate_prediction(model, trained_features, y_name, request.form)
            prediction_result = f"{float(prediction_result):.4f}" 

    except Exception as e:
        error = f"Gagal menghitung prediksi: {e}"
        flash(f"Error prediksi: {e}", 'danger')

    return render_template('prediksi.html', 
                           y_name=y_name,
                           categorical_mapping=FORM_CATEGORICAL_MAPPING,
                           numeric_var=NUMERIC_VAR,
                           metrics=metrics, 
                           prediction_result=prediction_result,
                           prediction_inputs=prediction_inputs,
                           error=error)
    

if __name__ == '__main__':
    app.run(debug=True)