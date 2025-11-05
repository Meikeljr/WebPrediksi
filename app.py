from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import pandas as pd
import statsmodels.api as sm
import os
import pickle
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'kunci_rahasia_dan_unik_anda' # Ganti dengan string yang kuat!

# --- FAKE USER DATABASE (Hanya untuk Contoh) ---
USERS = {
    "admin": {"password": "admin123", "id": 1, "name": "Admin Penjualan"}
}

# --- USER CLASS UNTUK FLASK-LOGIN ---
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

# --- KONFIGURASI DATA STATIS ---
STATIC_DATA_PATH = 'data/dummy.csv'
DETAIL_DATA_PATH = 'data/penjualandetail.csv'
DELIMITER = ',' 
CATEGORICAL_MAPPING = {
    "Periode": ["idul_fitri", "idul_adha"],
    "Jenis Kue": ["beng_beng", "kue_lainnya", "lidah_kucing", "nastar", "putri_salju", "rambutan", "sagu_keju", "semprit_susu"],
    "Ukuran": ["sedang", "kecil_besar"]
}
NUMERIC_VAR = "tahun"

# =======================================================
# UTILITY FUNCTIONS
# =======================================================

# Helper untuk menyimpan model ke session (encode/decode)
def encode_model(model):
    """Mengubah objek model menjadi string base64 yang aman untuk session."""
    return base64.b64encode(pickle.dumps(model)).decode('utf-8')

def decode_model(model_str):
    """Mengubah string base64 dari session kembali menjadi objek model."""
    return pickle.loads(base64.b64decode(model_str.encode('utf-8')))


def build_regression_model(data_path):
    data = pd.read_csv(data_path, sep=DELIMITER, header=0)
    Y_NAME = data.columns[0] 
    
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data.dropna(inplace=True)
    
    Y = data[Y_NAME]
    X = data.drop(columns=[Y_NAME])
    X = X.loc[:, (X != X.iloc[0]).any()] 
    
    if X.empty: raise ValueError("Model tidak dapat dibangun.")
    
    X = sm.add_constant(X, has_constant='add')
    model = sm.OLS(Y, X).fit()
    
    # Simpan model yang sudah di-fit ke dalam session
    session['fitted_model'] = encode_model(model)
    session['y_name'] = Y_NAME
    session['trained_features'] = X.columns.tolist()
    
    return model, Y_NAME, X.columns.tolist()

def calculate_prediction(model, trained_features, Y_NAME, form_data):
    input_data = pd.DataFrame(0, index=[0], columns=trained_features)
    input_data['const'] = 1
    
    tahun_val = pd.to_numeric(form_data.get(NUMERIC_VAR), errors='coerce')
    if tahun_val is None or pd.isna(tahun_val):
         raise ValueError("Input 'Tahun' harus berupa angka.")
    input_data[NUMERIC_VAR] = tahun_val
    
    for category_name, options in CATEGORICAL_MAPPING.items():
        selected_option = form_data.get(category_name)
        if selected_option and selected_option in trained_features:
             input_data[selected_option] = 1
        
    input_data = input_data[[col for col in input_data.columns if col in trained_features]]
    input_data = input_data[model.params.index]
    prediction = model.predict(input_data)[0]
    
    return prediction, input_data.drop(columns=['const']).to_dict('records')[0]

# =======================================================
# 1. ROUTES AUTENTIKASI (LOGIN & LOGOUT)
# =======================================================
# (Tidak Berubah dari Jawaban Sebelumnya)
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
    session.pop('fitted_model', None) # Hapus model dari session saat logout
    session.pop('y_name', None)
    session.pop('trained_features', None)
    flash('Anda telah berhasil logout.', 'success')
    return redirect(url_for('login'))

# =======================================================
# 2. ROUTES APLIKASI UTAMA
# =======================================================

@app.before_request
def check_model():
    """Jalankan model jika belum ada di session sebelum request prediksi/ringkasan."""
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
@login_required # Wajib login
def data_penjualan():
    try:
        # 1. Baca data detail baru dengan delimiter yang benar
        data = pd.read_csv(DETAIL_DATA_PATH, sep=DELIMITER, header=0)
        
        # 2. Hapus spasi dari nama kolom (untuk HTML/JS yang bersih)
        data.columns = [col.strip().replace(' ', '_') for col in data.columns]
        
        # 3. Ubah seluruh data menjadi HTML tanpa styling Pandas
        # classes='display' PENTING untuk DataTables
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
        
        # Ekstraksi Metrik
        metrics = {
            'R-squared': f"{model.rsquared:.4f}",
            'Adj. R-squared': f"{model.rsquared_adj:.4f}",
            'F-statistic': f"{model.fvalue:.4f}",
            'Prob (F-statistic)': f"{model.f_pvalue:.4f}",
            'N Observasi': model.nobs,
            'Jml Variabel Independen': len(model.params) - 1
        }
        
        # Persamaan Prediksi
        equation = f"{y_name} = {model.params['const']:.4f}"
        for col in model.params.index:
            if col != 'const':
                equation += f" + {model.params[col]:.4f} * {col.replace('_', ' ')}"

        # Tabel Koefisien
        if len(model.summary().tables) > 1:
            summary_table = model.summary().tables[1].as_html()
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
        # Jika model gagal dibangun di before_request, user akan diarahkan ke beranda
        return redirect(url_for('beranda'))
        
    try:
        model = decode_model(session['fitted_model'])
        y_name = session['y_name']
        trained_features = session['trained_features']
        
        # Ambil hanya metrik R-squared dan Adj. R-squared untuk ditampilkan di halaman prediksi
        metrics = {
            'R-squared': f"{model.rsquared:.4f}",
            'Adj. R-squared': f"{model.rsquared_adj:.4f}",
        }
        
        if request.method == 'POST':
            prediction_result, prediction_inputs = calculate_prediction(model, trained_features, y_name, request.form)
            prediction_result = f"{prediction_result:.4f}"

    except Exception as e:
        error = f"Gagal menghitung prediksi: {e}"
        flash(f"Error prediksi: {e}", 'danger')

    return render_template('prediksi.html', 
                           y_name=y_name,
                           categorical_mapping=CATEGORICAL_MAPPING,
                           numeric_var=NUMERIC_VAR,
                           metrics=metrics, 
                           prediction_result=prediction_result,
                           prediction_inputs=prediction_inputs,
                           error=error)
    

if __name__ == '__main__':
    app.run(debug=True)