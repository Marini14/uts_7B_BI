"""
UTS Business Intelligence
Integrasi Data Mining dan Business Intelligence pada Data Transaksi Toko

Universitas Timor

Menu:
  1. Upload Data (otomatis tersimpan & langsung lanjut ke menu 2)
  2. Exploratory Data Analysis (EDA)
  3. Data Preprocessing
  4. Apriori (Market Basket Analysis)
  5. K-Means (Segmentasi Pelanggan)
  6. Business Intelligence Dashboard
  7. Kesimpulan & Rekomendasi Keputusan Bisnis (a-f, dengan pembuktian)

Halaman Login tampil lebih dulu sebelum masuk ke menu utama.

Jalankan:  streamlit run Uts.py   (satu file)
"""

import hmac
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

KAMPUS = "Universitas Timor"
BASE_DIR = Path(__file__).resolve().parent

REQUIRED_COLS = [
    "Transaction_ID", "Transaction_Date", "Customer_ID", "Region",
    "Customer_Type", "Product_ID", "Product_Name", "Category",
    "Quantity", "Unit_Price", "Discount_Rate", "Line_Total", "Payment_Method",
]

MENU = [
    "1. Upload Data",
    "2. Exploratory Data Analysis",
    "3. Data Preprocessing",
    "4. Apriori (Market Basket)",
    "5. K-Means (Segmentasi)",
    "6. Business Intelligence Dashboard",
    "7. Kesimpulan & Rekomendasi",
]


# =====================================================================
# TEMA BIRU (latar & tampilan) + GRAFIK WARNA-WARNI
# =====================================================================
# Palet warna campur untuk grafik (tiap batang / irisan / titik beda warna)
PALETTE = ["#1565C0", "#E65100", "#2E7D32", "#6A1B9A", "#C62828",
           "#00838F", "#F9A825", "#AD1457", "#3949AB", "#5C6BC0",
           "#558B2F", "#00695C"]
MULTI_SCALE = px.colors.sequential.Turbo  # skala warna beragam untuk heatmap / gradasi

SEG_COLORS = {
    "Loyal (Bernilai Tinggi)": "#2E7D32",
    "Pelanggan Baru": "#1565C0",
    "Potensial": "#F9A825",
    "Berisiko Tidak Kembali": "#C62828",
}

CSS = """
<style>
:root {
    --b-bg1: #5AA9F5;
    --b-bg2: #8EC5FA;
    --b-card: #E3F0FF;
    --b-card2: #CFE5FF;
    --b-dark: #06214D;
    --b-mid: #0B5ED7;
    --b-deep: #0A3D91;
    --b-line: #1565C0;
}

/* ===== Background utama ===== */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: linear-gradient(160deg, var(--b-bg1) 0%, var(--b-bg2) 100%) !important;
    color: var(--b-dark);
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { background: transparent !important; }

/* ===== Sidebar ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A3D91 0%, #1565C0 100%) !important;
}
[data-testid="stSidebar"] * { color: #F2F8FF !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.35) !important; }

/* ===== Teks ===== */
h1, h2, h3, h4, h5, h6, p, label, li, span, div[data-testid="stMarkdownContainer"] {
    color: var(--b-dark);
}
h1, h2, h3 { color: #031B44 !important; font-weight: 800 !important; }
h3 { font-size: 1.15rem !important; }
[data-testid="stCaptionContainer"] * { color: #123B7A !important; }

/* ===== Banner judul halaman ===== */
.page-head {
    display: flex; align-items: center; gap: 14px;
    background: linear-gradient(90deg, #0A3D91 0%, #1976D2 100%);
    border-radius: 16px; padding: 14px 20px; margin: 0 0 16px;
    box-shadow: 0 6px 14px rgba(6, 33, 77, .28);
}
.page-ico { font-size: 30px; line-height: 1; }
.page-title { font-size: 25px; font-weight: 800; color: #FFFFFF !important; line-height: 1.2; }
.page-sub { font-size: 14px; color: #DCEBFF !important; margin-top: 2px; }

/* ===== Kartu halaman depan ===== */
.hero {
    background: linear-gradient(120deg, #0A3D91 0%, #1976D2 100%);
    border: 3px solid #E3F0FF;
    border-radius: 20px;
    padding: 22px 28px;
    margin-bottom: 18px;
    box-shadow: 0 6px 14px rgba(6, 33, 77, .3);
}
.hero-sub  { font-size: 14px; font-weight: 700; color: #BBD9FF !important; letter-spacing: .6px; text-transform: uppercase; }
.hero-name { font-size: 34px; font-weight: 800; color: #FFFFFF !important; line-height: 1.15; margin-top: 4px; }
.hero-desc { font-size: 16px; color: #E8F2FF !important; margin-top: 8px; max-width: 640px; }

/* ===== Identitas: logo (kiri) - teks - foto (kanan), ukuran sama ===== */
.id-row { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.id-badge, .id-photo {
    width: 48px; height: 48px; flex: none; border-radius: 50%;
    border: 2px solid #FFFFFF; box-sizing: border-box;
}
.id-badge { display: flex; align-items: center; justify-content: center; background: #FFFFFF; font-size: 24px; line-height: 1; }
.id-photo { object-fit: cover; object-position: 50% 30%; }
.login-head .id-badge, .login-head .id-photo { border-color: #1565C0; }
.id-mid { flex: 1; display: flex; align-items: center; justify-content: center; text-align: center; }
.side-brand { padding: 4px 0 2px; }
.side-campus { font-weight: 800; font-size: 16px; line-height: 1.2; color: #FFFFFF !important; }
.side-sub { font-size: 10.5px; font-weight: 700; letter-spacing: .4px; text-transform: uppercase; color: #E3F0FF !important; margin-top: 2px; }
.side-desc { font-size: 13.5px; font-weight: 600; line-height: 1.35; text-align: center; color: #FFFFFF !important; margin-top: 10px; }

/* ===== Metric card ===== */
[data-testid="stMetric"] {
    background: var(--b-card);
    border: 2px solid var(--b-line);
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 4px 10px rgba(6, 33, 77, .22);
}
[data-testid="stMetricLabel"] * { color: #123B7A !important; font-weight: 600; }
[data-testid="stMetricValue"] * { color: var(--b-deep) !important; font-weight: 800; }

/* ===== Tombol ===== */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
    background: var(--b-deep) !important;
    color: #F2F8FF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
    background: #062A66 !important;
    color: #FFFFFF !important;
}
.stButton > button *, .stDownloadButton > button *, [data-testid="stFormSubmitButton"] > button * { color: #F2F8FF !important; }
[data-testid="stFormSubmitButton"] > button { width: 100%; }

/* ===== Input ===== */
.stTextArea textarea, .stNumberInput input, .stTextInput input,
[data-baseweb="select"] > div, [data-baseweb="input"] {
    background: #EAF3FF !important;
    color: var(--b-dark) !important;
    border-color: var(--b-line) !important;
}
[data-testid="stFileUploader"] section {
    background: #EAF3FF !important;
    border: 2px dashed var(--b-deep) !important;
    border-radius: 12px;
}
[data-testid="stFileUploader"] section * { color: var(--b-dark) !important; }
[data-testid="stFileUploader"] button { background: var(--b-deep) !important; }
[data-testid="stFileUploader"] button * { color: #F2F8FF !important; }
[data-baseweb="tag"] { background: var(--b-mid) !important; }
[data-baseweb="tag"] * { color: #F2F8FF !important; }

/* ===== Slider ===== */
[data-testid="stSlider"] [role="slider"] { background: var(--b-deep) !important; }
[data-testid="stSlider"] div[data-testid="stTickBar"] * { color: var(--b-dark) !important; }

/* ===== Expander ===== */
[data-testid="stExpander"] {
    background: var(--b-card);
    border: 2px solid var(--b-line);
    border-radius: 12px;
}
[data-testid="stExpander"] summary * { color: var(--b-dark) !important; font-weight: 700; }

/* ===== Alert ===== */
[data-testid="stAlert"] {
    background: #E3F0FF !important;
    border-left: 6px solid var(--b-deep);
    border-radius: 10px;
}
[data-testid="stAlert"] * { color: var(--b-dark) !important; }

/* ===== Tabel ===== */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    background: var(--b-card);
    border: 2px solid var(--b-line);
    border-radius: 10px;
    padding: 2px;
}

/* ===== Chart plotly ===== */
.js-plotly-plot, .plotly, [data-testid="stPlotlyChart"] {
    background: rgba(255, 255, 255, .92) !important;
    border-radius: 14px;
    border: 2px solid var(--b-line);
}

/* ===== Radio menu sidebar ===== */
[data-testid="stSidebar"] [role="radiogroup"] label {
    background: rgba(255,255,255,.12);
    border-radius: 8px;
    padding: 6px 10px;
    margin-bottom: 4px;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(255,255,255,.28); }

/* ===== Halaman login (satu kartu, ringkas) ===== */
[data-testid="stForm"] {
    background: var(--b-card); border: 2px solid var(--b-line) !important; border-radius: 16px;
    padding: 14px 18px 10px; box-shadow: 0 6px 14px rgba(6, 33, 77, .25);
}
.login-head { text-align: center; margin-bottom: 8px; }
.login-sub { font-size: 11px; font-weight: 700; color: var(--b-deep); letter-spacing: .6px; text-transform: uppercase; }
.login-campus { font-size: 18px; font-weight: 800; color: #031B44; line-height: 1.2; }
.login-note { font-size: 12px; color: #123B7A; text-align: center; margin: 2px 0 6px; }

/* ===== Tata letak rapi ===== */
.block-container { padding-top: 2.2rem !important; max-width: 1300px !important; }
.sec-head {
    font-size: 20px; font-weight: 800; color: #031B44;
    background: linear-gradient(90deg, rgba(10,61,145,.28), rgba(10,61,145,0));
    border-left: 6px solid var(--b-deep); padding: 6px 12px; border-radius: 6px; margin: 26px 0 8px;
}
.sec-sub { font-size: 13.5px; color: #123B7A; margin: 0 0 10px 4px; }
.next-step {
    background: var(--b-card); border: 2px dashed var(--b-deep); border-radius: 12px;
    padding: 10px 14px; margin-top: 22px; color: var(--b-dark); font-size: 14px;
}
table.bi-table {
    width: 100%; border-collapse: collapse; background: #EAF3FF; border: 2px solid var(--b-line);
    font-size: 14px; margin: 6px 0 14px;
}
table.bi-table th { background: var(--b-mid); color: #F2F8FF !important; text-align: left; padding: 8px 10px; }
table.bi-table td { padding: 8px 10px; border-top: 1px solid rgba(21,101,192,.35); vertical-align: top; color: var(--b-dark); }
table.bi-table tr:nth-child(even) td { background: rgba(255,255,255,.65); }

/* ===== Scrollbar ===== */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: var(--b-deep); border-radius: 6px; }
</style>
"""


def apply_blue_theme():
    """Suntikkan CSS biru (latar & tampilan) dan set template Plotly dengan warna grafik campur."""
    st.markdown(CSS, unsafe_allow_html=True)
    pio.templates["blue"] = go.layout.Template(
        layout=dict(
            colorway=PALETTE,
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0.88)",
            font=dict(color="#06214D", size=13),
            title=dict(font=dict(color="#031B44", size=17)),
            xaxis=dict(gridcolor="rgba(21,101,192,.18)", linecolor="#1565C0", zerolinecolor="#1565C0"),
            yaxis=dict(gridcolor="rgba(21,101,192,.18)", linecolor="#1565C0", zerolinecolor="#1565C0"),
            legend=dict(bgcolor="rgba(255,255,255,.85)"),
            colorscale=dict(sequential=MULTI_SCALE, sequentialminus=MULTI_SCALE),
        )
    )
    pio.templates.default = "blue"
    px.defaults.template = "blue"
    px.defaults.color_discrete_sequence = PALETTE
    px.defaults.color_continuous_scale = MULTI_SCALE


# =====================================================================
# Helper tampilan (kompatibel dengan versi Streamlit lama & baru)
# =====================================================================
def _st_version():
    try:
        return tuple(int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        return (0, 0)


STRETCH = _st_version() >= (1, 50)


def df_show(data, **kw):
    if STRETCH:
        st.dataframe(data, width="stretch", **kw)
    else:
        st.dataframe(data, use_container_width=True, **kw)


def chart(fig, container=None, key=None):
    c = container if container is not None else st
    kw = {"key": key} if key else {}
    if STRETCH:
        c.plotly_chart(fig, width="stretch", **kw)
    else:
        c.plotly_chart(fig, use_container_width=True, **kw)


def bar_h(data, x, y, title):
    """Bar horizontal: setiap batang punya warna berbeda."""
    fig = px.bar(data, x=x, y=y, color=y, orientation="h", title=title,
                 color_discrete_sequence=PALETTE)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(categoryorder="total ascending")
    return fig


def line_color(data, x, y, title):
    """Garis tren dengan titik/marker berwarna-warni."""
    n = len(data)
    fig = go.Figure(go.Scatter(
        x=data[x], y=data[y], mode="lines+markers",
        line=dict(color="#0D47A1", width=2),
        marker=dict(size=12, color=[PALETTE[i % len(PALETTE)] for i in range(n)],
                    line=dict(color="#FFFFFF", width=1.5)),
    ))
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y, showlegend=False)
    return fig


def seg_map(segs):
    """Warna konsisten untuk tiap segmen pelanggan."""
    out, used = {}, set()
    for s in sorted(segs):
        if s in SEG_COLORS:
            out[s] = SEG_COLORS[s]
            used.add(SEG_COLORS[s])
    free = [c for c in PALETTE if c not in used]
    i = 0
    for s in sorted(segs):
        if s not in out:
            out[s] = free[i % len(free)]
            i += 1
    return out


def section(title, sub=None):
    """Judul bagian yang rapi (bilah warna + keterangan singkat)."""
    html = f'<div class="sec-head">{title}</div>'
    if sub:
        html += f'<div class="sec-sub">{sub}</div>'
    st.markdown(html, unsafe_allow_html=True)


def page_header(icon, title, sub=None):
    """Judul halaman berbentuk banner biru (seragam di semua menu)."""
    html = (f'<div class="page-head"><div class="page-ico">{icon}</div><div>'
            f'<div class="page-title">{title}</div>')
    if sub:
        html += f'<div class="page-sub">{sub}</div>'
    html += '</div></div>'
    st.markdown(html, unsafe_allow_html=True)


def next_step(text):
    """Petunjuk singkat langkah berikutnya di bagian bawah halaman."""
    st.markdown(f'<div class="next-step">➡️ <b>Langkah selanjutnya:</b> {text}</div>', unsafe_allow_html=True)


def show_hero():
    st.markdown(
        '<div class="hero">'
        '<div class="hero-sub">UTS Business Intelligence</div>'
        f'<div class="hero-name">{KAMPUS}</div>'
        '<div class="hero-desc">Integrasi Data Mining dan Business Intelligence '
        'pada Data Transaksi Toko</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# Foto profil kecil (JPEG 128x128, tertanam base64 agar tidak butuh file gambar terpisah)
PHOTO_B64 = "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"


def photo_uri():
    return "data:image/jpeg;base64," + PHOTO_B64


def id_row(mid_html):
    """Baris identitas: logo universitas (kiri), teks (tengah), foto (kanan) dengan ukuran sama."""
    return ('<div class="id-row"><div class="id-badge">🎓</div>'
            f'<div class="id-mid">{mid_html}</div>'
            f'<img class="id-photo" src="{photo_uri()}" alt="foto"></div>')


def show_sidebar_profile():
    st.sidebar.markdown(
        '<div class="side-brand">'
        + id_row(f'<div class="side-campus">{KAMPUS}</div>')
        + '<div class="side-desc">Integrasi Data Mining &amp; BI pada Data Transaksi Toko</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def scroll_top(token):
    """Gulung halaman ke paling atas setiap kali berpindah menu."""
    try:
        import streamlit.components.v1 as components
        js = """<script>
            const go = () => { try {
                const d = window.parent.document;
                ['section.main', '[data-testid="stMain"]', '[data-testid="stAppViewContainer"]'].forEach(s => {
                    const e = d.querySelector(s); if (e) { e.scrollTop = 0; }
                });
                d.documentElement.scrollTop = 0; d.body.scrollTop = 0;
            } catch (err) {} };
            go(); [80, 250, 600, 1200].forEach(t => setTimeout(go, t));
            </script>"""
        components.html(js + "<!-- " + str(token) + " -->", height=0)
    except Exception:
        pass


# =====================================================================
# LOGIN
# =====================================================================
# Daftar akun yang boleh masuk (username tidak peka huruf besar/kecil).
# Untuk menambah akun baru cukup tambahkan satu baris pada dictionary ini.
USERS = {
    "rini": {"password": "51230065", "tampil": "Rini"},
}


def check_login(username, password):
    """Kembalikan nama tampilan bila username & password benar, selain itu None."""
    u = USERS.get(str(username).strip().lower())
    if u is None:
        return None
    if hmac.compare_digest(str(password).encode(), u["password"].encode()):
        return u["tampil"]
    return None


def page_login():
    # sembunyikan sidebar selama belum login
    st.markdown(
        "<style>[data-testid=\"stSidebar\"], [data-testid=\"stSidebarCollapsedControl\"],"
        "[data-testid=\"collapsedControl\"] { display: none !important; }</style>",
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1.5, 1, 1.5])
    with mid:
        with st.form("login_form"):
            st.markdown(
                '<div class="login-head">'
                + id_row('<div class="login-campus">' + KAMPUS + '</div>')
                + '</div>',
                unsafe_allow_html=True,
            )
            username = st.text_input("Username", placeholder="Masukkan username")
            password = st.text_input("Password", type="password", placeholder="Masukkan password")
            st.markdown('<div class="login-note">Username: <b>Rini</b> · Password: <b>51230065</b></div>',
                        unsafe_allow_html=True)
            submit = st.form_submit_button("Masuk")
        if submit:
            nama = check_login(username, password)
            if nama:
                st.session_state["auth"] = True
                st.session_state["user"] = nama
                st.rerun()
            else:
                st.error("Username atau password salah. Silakan coba lagi.")


# =====================================================================
# Helper umum
# =====================================================================
def rupiah(x):
    if pd.isna(x):
        return "-"
    return "Rp " + f"{x:,.0f}".replace(",", ".")


def angka(x):
    return f"{x:,.0f}".replace(",", ".")


def need_raw():
    if st.session_state.get("raw") is None:
        st.warning("Belum ada data. Silakan upload file terlebih dahulu di menu **1. Upload Data**.")
        return False
    return True


def need_clean():
    if not need_raw():
        return False
    if st.session_state.get("clean") is None:
        st.warning("Data belum dibersihkan. Jalankan menu **3. Data Preprocessing** terlebih dahulu.")
        return False
    return True


def _clean_text(s):
    s = s.astype(str).str.strip()
    return s.where(~s.str.lower().isin(["nan", "none", "", "<na>", "nat"]), np.nan)


def _norm_by_mode(s):
    """Samakan variasi huruf/spasi: 'qris', 'Qris', 'QRIS ' -> bentuk yang paling sering muncul."""
    s = _clean_text(s)
    key = s.str.casefold()
    tmp = pd.DataFrame({"k": key, "v": s}).dropna()
    mode = tmp.groupby("k")["v"].agg(lambda x: x.value_counts().idxmax())
    return key.map(mode)


# =====================================================================
# Preprocessing (pure function, tidak menimpa Data_Raw)
# =====================================================================
def preprocess(raw, produk=None, pelanggan=None, max_qty=50, price_mult=3.0):
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    log = []
    n_awal = len(df)

    def add(masalah, jumlah, keputusan):
        log.append({"Masalah": masalah, "Jumlah Terdampak": int(jumlah), "Keputusan": keputusan})

    # 1. Rapikan teks (spasi) ------------------------------------------------
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]):
            df[c] = _clean_text(df[c])

    # 2. Format tanggal -----------------------------------------------------
    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")
    bad_date = df["Transaction_Date"].isna().sum()
    df = df.dropna(subset=["Transaction_Date", "Transaction_ID"])
    add("Tanggal tidak valid / Transaction_ID kosong", bad_date, "Baris dihapus")

    # 3. Duplikasi ----------------------------------------------------------
    n_dup = df.duplicated().sum()
    df = df.drop_duplicates().copy()
    add("Baris duplikat (seluruh kolom sama)", n_dup, "Baris dihapus (disimpan 1)")

    # 4. Inkonsistensi teks -------------------------------------------------
    for col in ["Category", "Payment_Method"]:
        before = df[col].nunique()
        raw_vals = df[col].copy()
        df[col] = _norm_by_mode(df[col])
        changed = ((raw_vals != df[col]) & raw_vals.notna()).sum()
        add(f"Inkonsistensi huruf/spasi pada {col} ({before} -> {df[col].nunique()} nilai unik)",
            changed, "Diperbaiki: disamakan ke ejaan yang paling sering muncul")

    # 5. Product_ID kosong --------------------------------------------------
    if produk is not None and {"Product_ID", "Product_Name"} <= set(produk.columns):
        name2id = produk.drop_duplicates("Product_Name").set_index("Product_Name")["Product_ID"]
    else:
        name2id = df.dropna(subset=["Product_ID"]).drop_duplicates("Product_Name").set_index("Product_Name")["Product_ID"]
    m = df["Product_ID"].isna()
    df.loc[m, "Product_ID"] = df.loc[m, "Product_Name"].map(name2id)
    add("Product_ID kosong", m.sum(), "Diimputasi dari Product_Name (katalog produk)")
    df = df.dropna(subset=["Product_ID"])

    # 6. Customer_ID kosong -> isi dari transaksi yang sama ------------------
    tid_cust = df.dropna(subset=["Customer_ID"]).groupby("Transaction_ID")["Customer_ID"].first()
    m = df["Customer_ID"].isna()
    df.loc[m, "Customer_ID"] = df.loc[m, "Transaction_ID"].map(tid_cust)
    sisa = df["Customer_ID"].isna().sum()
    add("Customer_ID kosong", m.sum(),
        f"Diisi dari baris lain pada Transaction_ID yang sama; {sisa} baris tetap kosong "
        "(dipertahankan untuk Apriori, dikeluarkan dari RFM/K-Means)")

    # 7. Region & Customer_Type kosong -> isi dari Data_Pelanggan -----------
    for col in ["Region", "Customer_Type"]:
        m = df[col].isna()
        if pelanggan is not None and col in pelanggan.columns and "Customer_ID" in pelanggan.columns:
            ref = pelanggan.drop_duplicates("Customer_ID").set_index("Customer_ID")[col]
            df.loc[m, col] = df.loc[m, "Customer_ID"].map(ref)
        sisa = df[col].isna().sum()
        df[col] = df[col].fillna("Tidak Diketahui")
        add(f"{col} kosong", m.sum(), f"Diisi dari Data_Pelanggan; sisa {sisa} diberi label 'Tidak Diketahui'")

    m = df["Payment_Method"].isna()
    df["Payment_Method"] = df["Payment_Method"].fillna("Tidak Diketahui")
    add("Payment_Method kosong", m.sum(), "Diberi label 'Tidak Diketahui'")

    # 8. Tipe numerik -------------------------------------------------------
    for c in ["Quantity", "Unit_Price", "Discount_Rate", "Line_Total"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Discount_Rate"] = df["Discount_Rate"].fillna(0).clip(0, 1)

    # 9. Quantity tidak valid ----------------------------------------------
    bad_q = df["Quantity"].isna() | (df["Quantity"] <= 0) | (df["Quantity"] > max_qty)
    add(f"Quantity kosong / <= 0 / > {max_qty}", bad_q.sum(), "Baris dihapus (tidak bisa dipercaya)")
    df = df[~bad_q].copy()
    df["Quantity"] = df["Quantity"].astype(int)

    # 10. Unit_Price tidak valid -------------------------------------------
    if produk is not None and {"Product_ID", "Reference_Unit_Price_Rp"} <= set(produk.columns):
        ref_price = produk.drop_duplicates("Product_ID").set_index("Product_ID")["Reference_Unit_Price_Rp"]
    else:
        ref_price = df[df["Unit_Price"] > 0].groupby("Product_ID")["Unit_Price"].median()
    ref_row = df["Product_ID"].map(ref_price)
    bad_p = df["Unit_Price"].isna() | (df["Unit_Price"] <= 0) | (df["Unit_Price"] > price_mult * ref_row)
    df.loc[bad_p, "Unit_Price"] = ref_row[bad_p]
    add(f"Unit_Price kosong / <= 0 / > {price_mult:g}x harga referensi", bad_p.sum(),
        "Diimputasi dengan harga referensi produk")
    df = df.dropna(subset=["Unit_Price"])

    # 11. Line_Total konsisten ---------------------------------------------
    expected = df["Quantity"] * df["Unit_Price"] * (1 - df["Discount_Rate"])
    bad_t = df["Line_Total"].isna() | ((df["Line_Total"] - expected).abs() > 1)
    df["Line_Total"] = np.where(bad_t, expected, df["Line_Total"]).round(0)
    add("Line_Total kosong / tidak sama dengan Qty x Harga x (1 - Diskon)", bad_t.sum(),
        "Dihitung ulang dari Quantity x Unit_Price x (1 - Discount_Rate)")

    # 12. Fitur turunan ----------------------------------------------------
    df["Month"] = df["Transaction_Date"].dt.to_period("M").dt.to_timestamp()
    df = df.reset_index(drop=True)

    add("RINGKASAN: jumlah baris sebelum -> sesudah", n_awal - len(df), f"{angka(n_awal)} -> {angka(len(df))} baris")
    return df, pd.DataFrame(log)


def build_rfm(df):
    d = df.dropna(subset=["Customer_ID"])
    ref_date = d["Transaction_Date"].max().normalize() + pd.Timedelta(days=1)
    rfm = d.groupby("Customer_ID").agg(
        Recency=("Transaction_Date", lambda x: (ref_date - x.max()).days),
        Frequency=("Transaction_ID", "nunique"),
        Monetary=("Line_Total", "sum"),
    ).reset_index()
    return rfm, ref_date


# =====================================================================
# Apriori & K-Means (dengan cache)
# =====================================================================
@st.cache_data(show_spinner=False)
def make_basket(df):
    b = df.groupby(["Transaction_ID", "Product_Name"]).size().unstack(fill_value=0) > 0
    return b


N_RULES = 25             # jumlah aturan asosiasi yang ditemukan dari data
SUPPORT_FLOOR = 0.001    # batas support terendah yang dicari
MIN_CONF = 0.05          # minimum confidence


@st.cache_data(show_spinner=False)
def run_apriori(df, min_support, min_conf):
    basket = make_basket(df)
    fi = apriori(basket, min_support=min_support * (1 - 1e-9), use_colnames=True)
    if fi.empty:
        return fi, pd.DataFrame()
    try:
        try:
            rules = association_rules(fi, num_itemsets=len(basket), metric="confidence", min_threshold=min_conf)
        except TypeError:
            rules = association_rules(fi, metric="confidence", min_threshold=min_conf)
    except Exception:
        rules = pd.DataFrame()
    if not rules.empty:
        rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
        rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
    fi = fi.copy()
    fi["itemsets"] = fi["itemsets"].apply(lambda s: ", ".join(sorted(s)))
    return fi, rules


@st.cache_data(show_spinner=False)
def evaluate_k(X, k_min, k_max):
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        rows.append({"k": k, "Inertia": km.inertia_, "Silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


STRATEGI = {
    "Loyal (Bernilai Tinggi)": "Program loyalitas/poin, akses awal produk baru, bundling premium, hindari diskon berlebihan.",
    "Pelanggan Baru": "Kampanye sambutan, voucher pembelian kedua, edukasi produk, ajak daftar member.",
    "Potensial": "Cross-selling berdasarkan aturan Apriori, diskon pembelian berikutnya, dorong kenaikan frekuensi.",
    "Berisiko Tidak Kembali": "Kampanye win-back (voucher terbatas waktu), pesan personal, survei alasan berhenti belanja.",
}


def label_clusters(prof):
    """prof: DataFrame index=cluster, kolom Recency/Frequency/Monetary (rata-rata)."""
    z = (prof - prof.mean()) / prof.std(ddof=0).replace(0, 1)
    score = -z["Recency"] + z["Frequency"] + z["Monetary"]
    order = score.sort_values(ascending=False).index.tolist()
    labels = {}
    for i, c in enumerate(order):
        if i == 0:
            labels[c] = "Loyal (Bernilai Tinggi)"
        elif i == len(order) - 1:
            labels[c] = "Berisiko Tidak Kembali"
        else:
            if prof.loc[c, "Frequency"] <= prof["Frequency"].median() and prof.loc[c, "Recency"] <= prof["Recency"].median():
                base = "Pelanggan Baru"
            else:
                base = "Potensial"
            labels[c] = base
    # beri akhiran bila ada label ganda
    totals = pd.Series(list(labels.values())).value_counts().to_dict()
    seen = {}
    for c in order:
        l = labels[c]
        seen[l] = seen.get(l, 0) + 1
        if totals[l] > 1:
            labels[c] = f"{l} {seen[l]}"
    return labels


# =====================================================================
# MENU 1 - UPLOAD DATA
# =====================================================================
def detect_sheet(xls):
    """Cari sheet transaksi: utamakan 'Data_Raw', lalu sheet mana pun yang berisi semua kolom wajib."""
    names = list(xls.sheet_names)
    names.sort(key=lambda s: 0 if s == "Data_Raw" else 1)
    for s in names:
        try:
            cols = [str(c).strip() for c in xls.parse(s, nrows=0).columns]
        except Exception:
            continue
        if all(c in cols for c in REQUIRED_COLS):
            return s
    return None


def page_upload():
    show_hero()
    page_header("📁", "Upload Data",
                "Unggah file data transaksi toko (Excel/CSV). Data otomatis tersimpan dan Anda "
                "langsung dibawa ke menu 2 (Exploratory Data Analysis).")

    f = st.file_uploader("Pilih file Excel (.xlsx) atau CSV", type=["xlsx", "xls", "csv"])

    if f is None:
        st.session_state["upload_sig"] = None
        if st.session_state.get("raw") is not None:
            raw = st.session_state["raw"]
            st.success(f"Data aktif: **{st.session_state['filename']}** "
                       f"({angka(len(raw))} baris, {raw.shape[1]} kolom). "
                       "Buka menu **2. Exploratory Data Analysis** untuk melanjutkan, "
                       "atau unggah file baru untuk mengganti data.")
            df_show(raw.head(20))
        else:
            st.info("Belum ada file yang diunggah.")
        return

    sig = (f.name, f.size)
    if st.session_state.get("upload_sig") == sig and st.session_state.get("raw") is not None:
        raw = st.session_state["raw"]
        st.success(f"File **{f.name}** sudah tersimpan ({angka(len(raw))} baris). "
                   "Pilih menu **2. Exploratory Data Analysis** untuk melanjutkan.")
        df_show(raw.head(20))
        return

    # ---------- file baru: baca -> validasi -> simpan otomatis ----------
    produk, pelanggan = None, None
    try:
        with st.spinner("Membaca file dan menyimpan data..."):
            if f.name.lower().endswith(".csv"):
                raw = pd.read_csv(f)
                sheet = "(CSV)"
            else:
                xls = pd.ExcelFile(f)
                sheet = detect_sheet(xls)
                if sheet is None:
                    st.error("Tidak ada sheet yang berisi semua kolom wajib. "
                             f"Sheet ditemukan: {', '.join(xls.sheet_names)}")
                    st.info("Kolom wajib: " + ", ".join(REQUIRED_COLS))
                    return
                raw = xls.parse(sheet)
                if "Data_Produk" in xls.sheet_names:
                    produk = xls.parse("Data_Produk")
                if "Data_Pelanggan" in xls.sheet_names:
                    pelanggan = xls.parse("Data_Pelanggan")
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return

    raw.columns = [str(c).strip() for c in raw.columns]
    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        st.error("Kolom wajib tidak ditemukan: " + ", ".join(missing))
        st.info("Pastikan file berisi data transaksi (mis. sheet `Data_Raw`).")
        return

    # simpan otomatis + reset hasil analisis sebelumnya
    st.session_state["raw"] = raw
    st.session_state["produk"] = produk
    st.session_state["pelanggan"] = pelanggan
    st.session_state["filename"] = f.name
    st.session_state["upload_sig"] = sig
    for k in ["clean", "log", "rfm", "ref_date", "rules", "rfm_seg"]:
        st.session_state[k] = None

    st.success(f"✅ File **{f.name}** (sheet **{sheet}**) berhasil dibaca dan otomatis tersimpan: "
               f"{angka(len(raw))} baris, {raw.shape[1]} kolom.")
    if produk is not None:
        st.info(f"Sheet katalog produk terdeteksi (`Data_Produk`, {len(produk)} produk).")
    if pelanggan is not None:
        st.info(f"Sheet pelanggan terdeteksi (`Data_Pelanggan`, {len(pelanggan)} pelanggan).")
    st.info("➡️ Data langsung dilanjutkan ke menu **2. Exploratory Data Analysis**...")
    st.subheader("Pratinjau data")
    df_show(raw.head(10))

    time.sleep(2.5)
    st.session_state["goto"] = MENU[1]
    st.rerun()


# =====================================================================
# MENU 2 - EDA
# =====================================================================
def page_eda():
    page_header("🔎", "Exploratory Data Analysis",
                "Semua dalam satu halaman: pertanyaan bisnis, ringkasan data, kualitas data, lalu visualisasi & insight.")
    if not need_raw():
        return
    raw = st.session_state["raw"].copy()
    raw["Transaction_Date"] = pd.to_datetime(raw["Transaction_Date"], errors="coerce")

    # ---------- 1. Business Understanding ----------
    section("1. Business Understanding", "Pertanyaan bisnis yang ingin dijawab melalui analisis ini.")
    default_q = (
        "1. Produk apa yang sering dibeli bersama?\n"
        "2. Segmen pelanggan mana yang paling bernilai?\n"
        "3. Kapan penjualan meningkat (pola bulanan)?"
    )
    st.text_area("Tuliskan minimal 3 pertanyaan bisnis (bisa diedit):", default_q, height=140)

    # ---------- 2. Ringkasan Data ----------
    section("2. Ringkasan Data", "Ukuran data, tipe kolom, dan statistik deskriptif.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah baris", angka(len(raw)))
    c2.metric("Jumlah kolom", raw.shape[1])
    c3.metric("Transaksi unik", angka(raw["Transaction_ID"].nunique()))
    c4.metric("Pelanggan unik", angka(raw["Customer_ID"].nunique()))
    c5, c6, c7 = st.columns(3)
    c5.metric("Produk unik", raw["Product_ID"].nunique())
    c6.metric("Total penjualan (sementara)", rupiah(pd.to_numeric(raw["Line_Total"], errors="coerce").sum()))
    c7.metric("Rentang tanggal",
              f"{raw['Transaction_Date'].min():%d %b %Y} - {raw['Transaction_Date'].max():%d %b %Y}")
    t1, t2 = st.columns([1, 2])
    with t1:
        st.subheader("Tipe data")
        df_show(raw.dtypes.astype(str).rename("Tipe").reset_index().rename(columns={"index": "Kolom"}))
    with t2:
        st.subheader("Statistik deskriptif numerik")
        num = raw[["Quantity", "Unit_Price", "Discount_Rate", "Line_Total"]].apply(pd.to_numeric, errors="coerce")
        df_show(num.describe().T)

    # ---------- 3. Kualitas Data ----------
    section("3. Kualitas Data", "Duplikasi, outlier, missing values, dan inkonsistensi kategori pada data mentah.")
    q = pd.to_numeric(raw["Quantity"], errors="coerce")
    p = pd.to_numeric(raw["Unit_Price"], errors="coerce")
    o1, o2, o3, o4, o5 = st.columns(5)
    o1.metric("Baris duplikat penuh", angka(raw.duplicated().sum()))
    o2.metric("Quantity <= 0", int((q <= 0).sum()))
    o3.metric("Quantity > 50", int((q > 50).sum()))
    o4.metric("Unit_Price <= 0", int((p <= 0).sum()))
    o5.metric("Unit_Price >= 100.000", int((p >= 100000).sum()))

    m_left, m_right = st.columns(2)
    with m_left:
        st.subheader("Missing values")
        mv = raw.isna().sum().rename("Jumlah Kosong").to_frame()
        mv["Persen"] = (mv["Jumlah Kosong"] / len(raw) * 100).round(2)
        df_show(mv)
    with m_right:
        st.subheader("Inkonsistensi kategori")
        for col in ["Category", "Payment_Method"]:
            vc = raw[col].astype(str).value_counts().rename("Jumlah").reset_index()
            vc.columns = [col, "Jumlah"]
            st.write(f"**{col}** ({raw[col].nunique()} nilai unik, sebelum dirapikan)")
            st.dataframe(vc, height=210)

    st.subheader("Outlier (boxplot data mentah)")
    bx = pd.DataFrame({
        "Kategori": raw["Category"].astype(str).str.strip().str.casefold().str.title().replace({"Atk": "ATK"}),
        "Quantity": q, "Unit_Price": p,
    })
    b1, b2 = st.columns(2)
    chart(px.box(bx, x="Kategori", y="Quantity", color="Kategori",
                 title="Boxplot Quantity per kategori (mentah)", log_y=True), b1)
    chart(px.box(bx, x="Kategori", y="Unit_Price", color="Kategori",
                 title="Boxplot Unit_Price per kategori (mentah)", log_y=True), b2)

    # ---------- 4. Visualisasi & Insight ----------
    section("4. Visualisasi & Insight", "Tren penjualan, kategori, metode pembayaran, dan insight awal.")
    d = raw.drop_duplicates().copy()
    d["Category"] = d["Category"].astype(str).str.strip().str.casefold().str.title()
    d["Category"] = d["Category"].replace({"Atk": "ATK"})
    d["Line_Total"] = pd.to_numeric(d["Line_Total"], errors="coerce")
    d = d[d["Line_Total"] < d["Line_Total"].quantile(0.999)]  # buang outlier ekstrem untuk visual
    d["Month"] = d["Transaction_Date"].dt.to_period("M").dt.to_timestamp()

    monthly = d.groupby("Month")["Line_Total"].sum().reset_index()
    chart(line_color(monthly, "Month", "Line_Total", "Tren penjualan bulanan"))

    v1, v2 = st.columns(2)
    cat = d.groupby("Category")["Line_Total"].sum().sort_values(ascending=False).reset_index()
    chart(bar_h(cat, "Line_Total", "Category", "Penjualan per kategori"), v1)
    pay = d["Payment_Method"].astype(str).str.strip().str.casefold().value_counts().reset_index()
    pay.columns = ["Metode", "Jumlah"]
    chart(px.pie(pay, names="Metode", values="Jumlah", title="Metode pembayaran"), v2)

    st.subheader("Insight awal (otomatis)")
    top_month = monthly.loc[monthly["Line_Total"].idxmax()]
    st.markdown(
        f"- Penjualan tertinggi terjadi pada **{top_month['Month']:%B %Y}** ({rupiah(top_month['Line_Total'])}).\n"
        f"- Kategori dengan penjualan terbesar: **{cat.iloc[0]['Category']}** "
        f"({cat.iloc[0]['Line_Total'] / cat['Line_Total'].sum():.1%} dari total).\n"
        f"- Metode pembayaran terbanyak: **{pay.iloc[0]['Metode']}**."
    )
    st.text_area("Tambahkan insight Anda sendiri (minimal 3):", height=120)
    next_step("buka menu **3. Data Preprocessing** untuk membersihkan data.")


# =====================================================================
# MENU 3 - PREPROCESSING
# =====================================================================
MV_AKSI = {
    "Transaction_ID": "Baris dihapus (ID transaksi tidak dapat diperkirakan)",
    "Transaction_Date": "Baris dihapus bila tanggal kosong / tidak valid",
    "Customer_ID": "Diisi dari baris lain pada Transaction_ID yang sama; sisanya dibiarkan kosong "
                   "(tetap dipakai Apriori, hanya dikeluarkan dari RFM/K-Means)",
    "Product_ID": "Diisi dari Product_Name (katalog produk)",
    "Region": "Diisi dari Data_Pelanggan; sisanya diberi label 'Tidak Diketahui'",
    "Customer_Type": "Diisi dari Data_Pelanggan; sisanya diberi label 'Tidak Diketahui'",
    "Payment_Method": "Diberi label 'Tidak Diketahui'",
    "Quantity": "Baris dihapus bila kosong / tidak valid",
    "Unit_Price": "Diisi dengan harga referensi produk",
    "Discount_Rate": "Diisi 0 (tanpa diskon)",
    "Line_Total": "Dihitung ulang dari Quantity x Unit_Price x (1 - Discount_Rate)",
}


def missing_report(raw, clean):
    """Tabel penjelasan: berapa nilai kosong per kolom dan diapakan (sebelum vs sesudah)."""
    r = raw.copy()
    r.columns = [str(c).strip() for c in r.columns]
    rows = []
    for c in REQUIRED_COLS:
        if c not in r.columns:
            continue
        s = r[c]
        if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
            s = _clean_text(s)
        before = int(s.isna().sum())
        after = int(clean[c].isna().sum()) if c in clean.columns else 0
        if before == 0 and after == 0:
            aksi = "Tidak ada nilai kosong"
        else:
            aksi = MV_AKSI.get(c, "Dibiarkan apa adanya")
        rows.append({"Kolom": c, "Kosong sebelum": before, "Kosong sesudah": after, "Penanganan": aksi})
    return pd.DataFrame(rows)


def page_preprocessing():
    page_header("🧹", "Data Preprocessing",
                "Pembersihan dilakukan pada salinan data kerja; Data_Raw tidak ditimpa.")
    if not need_raw():
        return

    hasil = st.container()   # ringkasan hasil + penanganan missing values tampil paling atas

    section("⚙️ Pengaturan pembersihan", "Atur batas kewajaran data, lalu jalankan preprocessing.")
    c1, c2 = st.columns(2)
    max_qty = c1.number_input("Batas maksimum Quantity yang dianggap wajar", 1, 10000, 50)
    price_mult = c2.number_input("Harga dianggap tidak wajar jika > N x harga referensi", 1.0, 100.0, 3.0)

    if st.button("▶️ Jalankan Preprocessing", type="primary"):
        with st.spinner("Membersihkan data..."):
            clean, log = preprocess(st.session_state["raw"], st.session_state.get("produk"),
                                    st.session_state.get("pelanggan"), max_qty, price_mult)
            rfm, ref_date = build_rfm(clean)
        st.session_state.update(clean=clean, log=log, rfm=rfm, ref_date=ref_date,
                                rules=None, rfm_seg=None)
        st.success("Preprocessing selesai.")

    if st.session_state.get("clean") is None:
        return

    clean, log, rfm = st.session_state["clean"], st.session_state["log"], st.session_state["rfm"]
    raw = st.session_state["raw"]

    # ---------- paling atas: ringkasan + penanganan missing values ----------
    with hasil:
        section("📊 Ringkasan hasil pembersihan")
        m1, m2, m3 = st.columns(3)
        m1.metric("Baris sebelum", angka(len(raw)))
        m2.metric("Baris sesudah", angka(len(clean)))
        m3.metric("Missing values tersisa", angka(int(clean.drop(columns=["Customer_ID"]).isna().sum().sum())))
        sisa_cust = int(clean["Customer_ID"].isna().sum())
        st.caption("Missing values tidak dibuang begitu saja: setiap kolom ditangani sesuai tabel di bawah "
                   "(diisi dari data lain, diberi label, atau dihitung ulang). "
                   f"Angka \"tersisa\" tidak menghitung Customer_ID; {angka(sisa_cust)} baris tanpa Customer_ID "
                   "sengaja dipertahankan untuk Apriori dan hanya dikeluarkan dari RFM/K-Means.")
        section("🧩 Penanganan Missing Values", "Apa yang dilakukan pada nilai kosong di setiap kolom.")
        df_show(missing_report(raw, clean), hide_index=True)

    section("📝 Log keputusan preprocessing", "Seluruh masalah data yang ditemukan dan keputusan perbaikannya.")
    df_show(log)

    section("🧺 basket_size per transaksi")
    basket_size = clean.groupby("Transaction_ID")["Product_ID"].nunique().rename("basket_size")
    st.write(f"Rata-rata basket_size: **{basket_size.mean():.2f}** produk unik per transaksi")
    bs = basket_size.value_counts().sort_index().reset_index()
    bs.columns = ["basket_size", "Jumlah Transaksi"]
    bs["basket_size"] = bs["basket_size"].astype(str)
    fig = px.bar(bs, x="basket_size", y="Jumlah Transaksi", color="basket_size",
                 title="Distribusi basket_size")
    fig.update_layout(showlegend=False)
    chart(fig)

    section("👤 Tabel RFM per pelanggan")
    st.write(f"Tanggal acuan Recency: **{st.session_state['ref_date']:%d %B %Y}** "
             "(1 hari setelah tanggal transaksi terakhir).")
    df_show(rfm.head(20))
    st.markdown(
        "**Catatan scaling:** K-Means memakai jarak Euclidean sehingga fitur dengan skala besar "
        "(Monetary dalam Rupiah) akan mendominasi. Karena itu RFM distandardisasi (StandardScaler) "
        "sebelum clustering. Untuk data besar dapat digunakan sampling, chunking, atau agregasi "
        "per pelanggan (seperti tabel RFM ini)."
    )

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Unduh data bersih (CSV)", clean.to_csv(index=False).encode("utf-8"),
                       "data_clean.csv", "text/csv")
    d2.download_button("⬇️ Unduh tabel RFM (CSV)", rfm.to_csv(index=False).encode("utf-8"),
                       "rfm.csv", "text/csv")
    next_step("buka menu **4. Apriori (Market Basket)**.")


# =====================================================================
# MENU 4 - APRIORI (parameter ditentukan otomatis dari data)
# =====================================================================
def _assoc(fi, n, min_conf):
    try:
        return association_rules(fi, num_itemsets=n, metric="confidence", min_threshold=min_conf)
    except TypeError:
        return association_rules(fi, metric="confidence", min_threshold=min_conf)


@st.cache_data(show_spinner=False)
def auto_apriori_params(df, n_rules=N_RULES):
    """Cari (min_support, min_confidence) otomatis dari data.

    - min_confidence = MIN_CONF (0.05) dan min_lift = 1 (asosiasi positif).
    - min_support = support aturan ke-`n_rules` (terbesar), sehingga aturan yang ditemukan berjumlah 25.
    """
    basket = make_basket(df)
    n = len(basket)
    fi = apriori(basket, min_support=SUPPORT_FLOOR, use_colnames=True)
    if fi.empty:
        return SUPPORT_FLOOR, MIN_CONF
    try:
        r = _assoc(fi, n, MIN_CONF)
        r = r[r["lift"] >= 1.0] if not r.empty else r
    except Exception:
        return SUPPORT_FLOOR, MIN_CONF
    if r.empty:
        return SUPPORT_FLOOR, MIN_CONF
    sups = r["support"].sort_values(ascending=False).to_numpy()
    s = float(sups[min(n_rules, len(sups)) - 1])
    return s, MIN_CONF


def get_rules(df, min_support, min_conf, min_lift=1.0, cap=None):
    fi, rules = run_apriori(df, min_support, min_conf)
    if not rules.empty:
        rules = (rules[rules["lift"] >= min_lift]
                 .sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True))
        if cap:
            rules = rules.head(cap).reset_index(drop=True)
    return fi, rules


def dedup_rules(rules):
    """Gabungkan aturan cermin (A->B dan B->A) agar satu pasangan produk hanya muncul sekali."""
    if rules is None or rules.empty:
        return rules
    key = rules.apply(lambda r: frozenset(r["antecedents"].split(", ") + r["consequents"].split(", ")), axis=1)
    return rules[~key.duplicated()].reset_index(drop=True)


def page_apriori():
    page_header("🛒", "Apriori - Market Basket Analysis",
                "Temukan produk yang sering dibeli bersamaan untuk bundling dan penempatan produk.")
    if not need_clean():
        return
    clean = st.session_state["clean"]
    basket = make_basket(clean)
    n_trx = basket.shape[0]
    st.write(f"Matriks transaksi x produk: **{angka(n_trx)} transaksi** x **{basket.shape[1]} produk** "
             "(1 jika produk dibeli, 0 jika tidak; satu basket = produk unik per Transaction_ID).")

    with st.spinner("Menentukan parameter Apriori otomatis dari data..."):
        auto_s, auto_c = auto_apriori_params(clean)

    section("⚙️ Parameter Apriori",
            f"Nilai awal ditentukan otomatis dari data sehingga ditemukan {N_RULES} aturan asosiasi. "
            "Geser slider jika ingin mengatur parameter secara manual.")
    sig = f"{auto_s:.8f}"
    s_max = max(0.100, float(np.ceil(auto_s * 1000) / 1000))
    c1, c2, c3 = st.columns(3)
    min_support = c1.slider("Minimum support", 0.001, s_max, float(auto_s), 0.0005,
                            format="%.4f", key=f"ap_s_{sig}")
    min_conf = c2.slider("Minimum confidence", 0.05, 1.0, float(auto_c), 0.05, key=f"ap_c_{sig}")
    min_lift = c3.slider("Minimum lift", 1.0, 3.0, 1.0, 0.05, key=f"ap_l_{sig}")
    is_auto = (abs(min_support - auto_s) < 1e-9 and abs(min_conf - auto_c) < 1e-9
               and abs(min_lift - 1.0) < 1e-9)

    fi, rules = get_rules(clean, min_support, min_conf, min_lift, cap=N_RULES if is_auto else None)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Minimum support", f"{min_support:.4f}")
    m2.metric("Minimum confidence", f"{min_conf:.2f}")
    m3.metric("Minimum lift", f"{min_lift:.2f}")
    m4.metric("Frequent itemsets", angka(len(fi)))
    m5.metric("Aturan ditemukan", angka(len(rules)))
    if is_auto and 0 < len(rules) < N_RULES:
        st.info(f"Data hanya menghasilkan {len(rules)} aturan pada confidence ≥ {MIN_CONF:.2f} dan lift ≥ 1.")

    # ---------- Frequent itemsets (langsung tampil, satu kali) ----------
    section("📋 Frequent Itemsets", f"Itemset dengan support ≥ {min_support:.4f} (urut dari support terbesar).")
    if fi.empty:
        st.warning("Tidak ada frequent itemset pada parameter ini. Turunkan minimum support.")
    else:
        fi_show = fi.sort_values("support", ascending=False).copy()
        fi_show["Jumlah item"] = fi_show["itemsets"].str.count(", ") + 1
        fi_show["Jumlah transaksi"] = (fi_show["support"] * n_trx).round().astype(int)
        fi_show = fi_show.rename(columns={"itemsets": "Itemset", "support": "Support"})
        df_show(fi_show[["Itemset", "Jumlah item", "Support", "Jumlah transaksi"]].round(4),
                height=320, hide_index=True)
        st.caption(f"Total frequent itemsets: {angka(len(fi))}")

    if rules.empty:
        st.warning("Tidak ada aturan asosiasi pada parameter ini. Turunkan support / confidence.")
        st.session_state["rules"] = None
        return
    st.session_state["rules"] = rules
    st.session_state["apriori_params"] = (min_support, min_conf, min_lift)

    # ---------- aturan terbaik ----------
    top = rules.head(N_RULES).copy()
    section(f"🏆 {min(N_RULES, len(rules))} Aturan Asosiasi Terbaik",
            "Pemeringkatan: lift tertinggi (kekuatan asosiasi di atas kebetulan), lalu confidence.")
    tab = top[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
    tab.insert(0, "No", range(1, len(tab) + 1))
    tab["Jumlah transaksi"] = (tab["support"] * n_trx).round().astype(int)
    tab.columns = ["No", "Jika membeli", "Maka juga membeli", "Support", "Confidence", "Lift", "Jumlah transaksi"]
    df_show(tab.round(4), hide_index=True, height=560)

    section("📊 Visualisasi aturan")
    v1, v2 = st.columns(2)
    chart(px.scatter(top, x="support", y="confidence", color="lift", size="lift",
                     hover_data=["antecedents", "consequents"],
                     title="Support vs Confidence (warna = lift)"), v1)
    lab = top.head(10).copy()
    lab["Aturan"] = lab["antecedents"] + " → " + lab["consequents"]
    chart(bar_h(lab, "lift", "Aturan", "10 aturan dengan lift tertinggi"), v2)

    section("💡 Rekomendasi cross-selling / bundling (hipotesis)")
    for _, r in dedup_rules(rules).head(3).iterrows():
        st.markdown(
            f"- **Bundling {r['antecedents']} + {r['consequents']}**: confidence {r['confidence']:.1%}, "
            f"lift {r['lift']:.2f}. Letakkan kedua produk berdekatan di rak dan uji paket promo."
        )
    st.info("Aturan asosiasi bukan bukti sebab-akibat. Dampak penjualan harus diuji (mis. A/B test) sebelum dianggap fakta.")
    next_step("buka menu **5. K-Means (Segmentasi)**.")


# =====================================================================
# MENU 5 - K-MEANS (Elbow dari data -> pilih k -> Elbow terbaru -> hasil)
# =====================================================================
K_MAX = 6    # k dievaluasi dari 2 sampai 6 (pilihan: 3, 4, 5, dst.)


def scale_rfm(rfm, use_log=True):
    feats = rfm[["Recency", "Frequency", "Monetary"]].astype(float)
    X = np.log1p(feats) if use_log else feats
    return StandardScaler().fit_transform(X)


def elbow_k(ev):
    """Titik siku: titik kurva inertia yang paling jauh dari garis lurus antara titik awal & akhir."""
    k = ev["k"].to_numpy(float)
    y = ev["Inertia"].to_numpy(float)
    if len(k) < 3 or y.max() == y.min():
        return int(k[0])
    kn = (k - k.min()) / (k.max() - k.min())
    yn = (y - y.min()) / (y.max() - y.min())
    return int(k[int(np.argmax(1 - kn - yn))])


def fit_segments(rfm, X, k):
    rfm = rfm.copy()
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rfm["Cluster"] = km.labels_
    labels = label_clusters(rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean())
    rfm["Segmen"] = rfm["Cluster"].map(labels)
    return rfm, km


def auto_segment(rfm, use_log=True):
    X = scale_rfm(rfm, use_log)
    ev = evaluate_k(X, 2, max(3, min(K_MAX, len(rfm) - 1)))
    k = elbow_k(ev)
    seg, _ = fit_segments(rfm, X, k)
    return seg, k


def elbow_fig(ev, k_elbow):
    fig = line_color(ev, "k", "Inertia", "Elbow Method (titik siku ditandai)")
    row = ev[ev["k"] == k_elbow].iloc[0]
    fig.add_trace(go.Scatter(
        x=[k_elbow], y=[row["Inertia"]], mode="markers+text", text=[f"Elbow k = {k_elbow}"],
        textposition="top right", showlegend=False,
        marker=dict(size=24, color="rgba(0,0,0,0)", line=dict(color="#C62828", width=4))))
    fig.update_xaxes(dtick=1)
    return fig


def elbow_fig_pilih(ev, k, k_elbow):
    """Elbow terbaru: garis solid k=2 s.d. k terpilih, garis putus-putus setelahnya, k terpilih dilingkari."""
    sd = ev[ev["k"] <= k]
    sisa = ev[ev["k"] >= k]
    row = ev[ev["k"] == k].iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sisa["k"], y=sisa["Inertia"], mode="lines+markers", name="k setelah k terpilih",
        line=dict(color="#F9A825", dash="dash", width=2), marker=dict(size=9)))
    fig.add_trace(go.Scatter(
        x=sd["k"], y=sd["Inertia"], mode="lines+markers", name=f"k = 2 s.d. {k}",
        line=dict(color="#2E7D32", width=3), marker=dict(size=12)))
    fig.add_trace(go.Scatter(
        x=[k], y=[row["Inertia"]], mode="markers+text", text=[f"k terpilih = {k}"],
        textposition="top right", name="k terpilih",
        marker=dict(size=24, color="rgba(0,0,0,0)", line=dict(color="#C62828", width=4))))
    fig.add_vline(x=k, line_dash="dash", line_color="#C62828")
    fig.update_layout(title=f"Elbow Method terbaru (k terpilih = {k})",
                      xaxis_title="k", yaxis_title="Inertia")
    fig.update_xaxes(dtick=1)
    return fig


def silhouette_fig(ev, k_sil):
    fig = line_color(ev, "k", "Silhouette", "Silhouette Score")
    row = ev[ev["k"] == k_sil].iloc[0]
    fig.add_trace(go.Scatter(
        x=[k_sil], y=[row["Silhouette"]], mode="markers+text", text=[f"Tertinggi k = {k_sil}"],
        textposition="top right", showlegend=False,
        marker=dict(size=24, color="rgba(0,0,0,0)", line=dict(color="#2E7D32", width=4))))
    fig.update_xaxes(dtick=1)
    return fig


def kmeans_result(rfm, X, ev, k, k_elbow, k_sil, judul, tag):
    """Tampilkan hasil K-Means untuk satu nilai k; kembalikan data RFM + segmen."""
    rfm, km = fit_segments(rfm, X, k)
    sil = float(ev.loc[ev["k"] == k, "Silhouette"].iloc[0])
    section(judul)
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Jumlah klaster (k)", k)
    h2.metric("Inertia", angka(km.inertia_))
    h3.metric("Silhouette", f"{sil:.3f}")
    h4.metric("Pelanggan tersegmentasi", angka(len(rfm)))
    if k != k_elbow:
        s_el = float(ev.loc[ev["k"] == k_elbow, "Silhouette"].iloc[0])
        st.caption(f"Perbandingan: k titik elbow = {k_elbow} memiliki silhouette {s_el:.3f}; "
                   f"pilihan k = {k} memiliki silhouette {sil:.3f}.")

    st.markdown("**Profil klaster**")
    prof = rfm.groupby(["Cluster", "Segmen"]).agg(
        Jumlah=("Customer_ID", "count"),
        Recency_Mean=("Recency", "mean"), Recency_Median=("Recency", "median"),
        Frequency_Mean=("Frequency", "mean"), Frequency_Median=("Frequency", "median"),
        Monetary_Mean=("Monetary", "mean"), Monetary_Median=("Monetary", "median"),
    ).round(1).reset_index()
    df_show(prof, hide_index=True)

    cmap = seg_map(rfm["Segmen"].unique())
    s1, s2 = st.columns(2)
    chart(px.scatter(rfm, x="Frequency", y="Monetary", color="Segmen", opacity=0.75,
                     color_discrete_map=cmap, title="Scatter Frequency vs Monetary"), s1, key=f"km_{tag}_sc1")
    chart(px.scatter(rfm, x="Recency", y="Monetary", color="Segmen", opacity=0.75,
                     color_discrete_map=cmap, title="Scatter Recency vs Monetary"), s2, key=f"km_{tag}_sc2")
    comp = prof.groupby("Segmen")["Jumlah"].sum().reset_index()
    avg_m = rfm.groupby("Segmen")["Monetary"].mean().reset_index()
    g1, g2 = st.columns(2)
    chart(px.pie(comp, names="Segmen", values="Jumlah", color="Segmen",
                 color_discrete_map=cmap, title="Komposisi pelanggan per segmen"), g1, key=f"km_{tag}_pie")
    chart(px.bar(avg_m, x="Segmen", y="Monetary", color="Segmen", color_discrete_map=cmap,
                 title="Rata-rata Monetary per segmen").update_layout(showlegend=False), g2, key=f"km_{tag}_bar")

    st.markdown("**🎯 Strategi pemasaran per segmen**")
    for seg in sorted(rfm["Segmen"].unique()):
        base = seg.rsplit(" ", 1)[0] if seg[-1].isdigit() else seg
        st.markdown(f"- **{seg}**: {STRATEGI.get(base, '-')}")
    st.caption("Label dibuat otomatis dari profil RFM; verifikasi kembali dengan tabel profil di atas.")
    return rfm


def page_kmeans():
    page_header("👥", "K-Means - Segmentasi Pelanggan",
                "Kelompokkan pelanggan berdasarkan Recency, Frequency, dan Monetary (RFM).")
    if not need_clean():
        return
    rfm = st.session_state["rfm"].copy()
    if len(rfm) < 4:
        st.warning("Jumlah pelanggan terlalu sedikit untuk segmentasi K-Means.")
        return
    st.write(f"Jumlah pelanggan (1 baris = 1 pelanggan): **{angka(len(rfm))}**")
    use_log = st.checkbox("Transformasi log1p sebelum scaling (mengurangi skewness)", value=True)
    X = scale_rfm(rfm, use_log)
    k_max = max(3, min(K_MAX, len(rfm) - 1))

    # ---------- A. Grafik Elbow dari data ----------
    ev = evaluate_k(X, 2, k_max)
    k_elbow = elbow_k(ev)
    k_sil = int(ev.loc[ev["Silhouette"].idxmax(), "k"])

    section("📈 A. Grafik Elbow dari data",
            f"Dihitung otomatis untuk k = 2 sampai {k_max}. Titik elbow = k setelah itu penambahan klaster "
            "tidak lagi menurunkan inertia secara berarti. Grafik ini tidak berubah saat Anda memilih k.")
    e1, e2 = st.columns(2)
    chart(elbow_fig(ev, k_elbow), e1)
    chart(silhouette_fig(ev, k_sil), e2)
    a1, a2 = st.columns(2)
    a1.metric("Titik elbow dari data", k_elbow)
    a2.metric("k dengan silhouette tertinggi", k_sil)

    # ---------- B. Hasil K-Means dengan k dari data (elbow) ----------
    rfm_e = kmeans_result(rfm, X, ev, k_elbow, k_elbow, k_sil,
                          f"✅ B. Hasil K-Means dengan k = {k_elbow} (titik elbow dari data)", "elbow")
    st.session_state["rfm_seg"] = rfm_e
    st.session_state["k_used"] = k_elbow

    # ---------- C. Penentuan k secara manual ----------
    section("🎯 C. Tentukan jumlah klaster (k) secara manual",
            "Pilih k yang diinginkan (mis. 3, 4, atau 5). Grafik Elbow terbaru dan hasilnya tampil di bawah.")
    opsi = [int(v) for v in ev["k"]]
    k = st.radio("Jumlah klaster (k)", opsi, index=opsi.index(k_elbow), horizontal=True,
                 key=f"km_k_{k_elbow}_{k_max}_{len(rfm)}_{int(use_log)}")
    if k == k_elbow:
        st.success(f"k = **{k}** sama dengan titik elbow dari data.")
    else:
        st.info(f"k = **{k}** dipilih manual (titik elbow dari data: k = {k_elbow}).")

    # ---------- D. Elbow terbaru ----------
    section(f"📉 D. Grafik Elbow terbaru (k = {k})")
    chart(elbow_fig_pilih(ev, k, k_elbow))
    ev_t = ev.copy()
    ev_t["Penurunan Inertia (%)"] = -ev_t["Inertia"].pct_change() * 100
    ev_t["Keterangan"] = ""
    ev_t.loc[ev_t["k"] == k_elbow, "Keterangan"] += "Titik elbow (data)  "
    ev_t.loc[ev_t["k"] == k_sil, "Keterangan"] += "Silhouette tertinggi  "
    ev_t.loc[ev_t["k"] == k, "Keterangan"] += "◀ dipakai"
    df_show(ev_t.round(4), hide_index=True)

    # ---------- E. Hasil untuk k pilihan ----------
    if k == k_elbow:
        st.info(f"Hasil untuk k = {k} sama dengan bagian B di atas. Pilih k lain pada bagian C untuk membandingkan.")
    else:
        rfm_k = kmeans_result(rfm, X, ev, k, k_elbow, k_sil,
                              f"✅ E. Hasil K-Means dengan k = {k} (pilihan Anda)", "pilih")
        st.session_state["rfm_seg"] = rfm_k
        st.session_state["k_used"] = k
    next_step("buka menu **6. Business Intelligence Dashboard**.")


# =====================================================================
# MENU 6 - DASHBOARD BI
# =====================================================================
def page_dashboard():
    page_header("📊", "Business Intelligence Dashboard",
                "Ringkasan kinerja penjualan, hasil Apriori, dan profil segmen pelanggan (K-Means).")
    if not need_clean():
        return
    d = st.session_state["clean"]
    seg = st.session_state.get("rfm_seg")

    basket_size = d.groupby("Transaction_ID")["Product_ID"].nunique()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total penjualan", rupiah(d["Line_Total"].sum()))
    k2.metric("Jumlah transaksi", angka(d["Transaction_ID"].nunique()))
    k3.metric("Pelanggan unik", angka(d["Customer_ID"].nunique()))
    k4.metric("Rata-rata basket_size", f"{basket_size.mean():.2f}")

    section("📈 Tren penjualan")
    monthly = d.groupby("Month")["Line_Total"].sum().reset_index()
    chart(line_color(monthly, "Month", "Line_Total", "Tren penjualan bulanan"))

    section("🏷️ Produk, kategori, wilayah, dan pembayaran")
    t1, t2 = st.columns(2)
    top_p = d.groupby("Product_Name")["Line_Total"].sum().nlargest(10).reset_index()
    chart(bar_h(top_p, "Line_Total", "Product_Name", "Top 10 produk (penjualan)"), t1)
    top_c = d.groupby("Category")["Line_Total"].sum().reset_index()
    chart(bar_h(top_c, "Line_Total", "Category", "Penjualan per kategori"), t2)

    r1, r2 = st.columns(2)
    reg = d.groupby("Region")["Line_Total"].sum().reset_index()
    chart(bar_h(reg, "Line_Total", "Region", "Penjualan per wilayah"), r1)
    pay = d["Payment_Method"].value_counts().reset_index()
    pay.columns = ["Metode", "Jumlah"]
    chart(px.pie(pay, names="Metode", values="Jumlah", title="Metode pembayaran"), r2)

    section("🛒 Hasil Apriori", "10 aturan asosiasi teratas.")
    rules = st.session_state.get("rules")
    if rules is not None and not rules.empty:
        top = rules.head(10)[["antecedents", "consequents", "support", "confidence", "lift"]].round(4)
        top.columns = ["Jika membeli", "Maka juga membeli", "Support", "Confidence", "Lift"]
        df_show(top, hide_index=True)
    else:
        st.info("Belum ada hasil Apriori. Jalankan menu **4. Apriori** terlebih dahulu.")

    section("👥 Profil K-Means", "Ringkasan rata-rata RFM per segmen pelanggan.")
    if seg is not None:
        prof = seg.groupby("Segmen").agg(
            Jumlah=("Customer_ID", "count"), Recency=("Recency", "mean"),
            Frequency=("Frequency", "mean"), Monetary=("Monetary", "mean")).round(1).reset_index()
        cmap = seg_map(prof["Segmen"].unique())
        p1, p2 = st.columns([2, 1])
        with p1:
            df_show(prof, hide_index=True)
        chart(px.pie(prof, names="Segmen", values="Jumlah", color="Segmen",
                     color_discrete_map=cmap, title="Komposisi segmen"), p2)

        st.markdown("**Bonus: pola kategori produk per segmen** (nilai penjualan)")
        m = d.merge(seg[["Customer_ID", "Segmen"]], on="Customer_ID", how="inner")
        if not m.empty:
            pv = m.pivot_table(index="Segmen", columns="Category", values="Line_Total", aggfunc="sum", fill_value=0)
            chart(px.imshow(pv, aspect="auto", color_continuous_scale=MULTI_SCALE))
    else:
        st.info("Belum ada hasil K-Means. Jalankan menu **5. K-Means** terlebih dahulu.")
    next_step("buka menu **7. Kesimpulan & Rekomendasi**.")


# =====================================================================
# MENU 7 - KESIMPULAN & REKOMENDASI KEPUTUSAN BISNIS (Tahap 6 / Soal 10)
# =====================================================================
ASSUMP = {
    "as_conv": 5.0,       # % transaksi terpapar yang menambah produk pasangan
    "as_loyal": 5.0,      # % kenaikan belanja segmen loyal
    "as_react": 10.0,     # % pelanggan berisiko yang kembali aktif
    "as_newfreq": 10.0,   # % kenaikan frekuensi pelanggan baru / potensial
    "as_season": 10.0,    # % kenaikan penjualan pada bulan lemah
    "as_promo": 10.0,     # biaya promosi/diskon (% dari tambahan penjualan)
    "as_voucher": 10000,  # nilai voucher (Rp) per transaksi tambahan / pelanggan kembali
}


def asm(key):
    return st.session_state.get(key, ASSUMP[key])


def _pct(x):
    return f"{x:.1%}"


def _fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "-"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def months_span(d):
    return max(1.0, (d["Transaction_Date"].max() - d["Transaction_Date"].min()).days / 30.44)


def segment_stats(seg):
    sg = seg.groupby("Segmen").agg(Jumlah=("Customer_ID", "count"), Nilai=("Monetary", "sum"),
                                   Recency=("Recency", "mean"), Frequency=("Frequency", "mean"),
                                   Monetary=("Monetary", "mean"))
    sg["pc"] = sg["Jumlah"] / sg["Jumlah"].sum()
    sg["pv"] = sg["Nilai"] / sg["Nilai"].sum()
    return sg.sort_values("pv", ascending=False)


def seg_groups(sg):
    loyal = next((s for s in sg.index if s.startswith("Loyal")), None)
    risk = next((s for s in sg.index if s.startswith("Berisiko")), None)
    mid = [s for s in sg.index if s.startswith("Potensial") or s.startswith("Pelanggan Baru")]
    return loyal, risk, mid


def html_table(df):
    html = df.to_html(index=False, classes="bi-table", border=0, escape=True)
    st.markdown(html.replace("\n", ""), unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Estimasi dampak (d) + penilaian prioritas (c)
# ---------------------------------------------------------------------
def estimate_impact(d, seg, rules, basket, A):
    conv, upl, react = A["as_conv"] / 100, A["as_loyal"] / 100, A["as_react"] / 100
    nfreq, season, promo, voucher = A["as_newfreq"] / 100, A["as_season"] / 100, A["as_promo"] / 100, A["as_voucher"]
    months = months_span(d)
    sg = segment_stats(seg)
    loyal, risk, mid = seg_groups(sg)
    monthly = d.groupby("Month")["Line_Total"].sum()
    prod_avg = d.groupby("Product_Name")["Line_Total"].mean()
    items, extra_items, exposed = [], 0.0, 0

    if rules is not None and not rules.empty:
        rev = 0.0
        for _, r in rules.head(3).iterrows():
            Ai, Bi = r["antecedents"].split(", "), r["consequents"].split(", ")
            if any(x not in basket.columns for x in Ai + Bi):
                continue
            a, b = basket[Ai].all(axis=1), basket[Bi].all(axis=1)
            gap = int((a & ~b).sum())
            rev += gap * conv * sum(prod_avg.get(x, 0.0) for x in Bi)
            extra_items += gap * conv * len(Bi)
            exposed += gap
        items.append(dict(
            nama="Bundling & penempatan produk (3 aturan Apriori teratas)", rev=rev, cost=rev * promo,
            urg=3.0, mudah=4.0,
            alasan="Peluang tambahan penjualan; tidak ada risiko kehilangan pelanggan sehingga urgensi sedang.",
            asumsi=f"{A['as_conv']:.1f}% dari {angka(exposed)} transaksi yang membeli produk pemicu tetapi belum membeli "
                   f"pasangannya akhirnya menambah pasangan produk; biaya diskon {A['as_promo']:.0f}% dari tambahan penjualan.",
            rumus="Transaksi terpapar x konversi x rata-rata nilai produk pasangan",
            kpi="Attach rate pasangan produk; basket_size; nilai per transaksi",
            ukur="A/B test 4-6 minggu: hari/toko dengan bundling vs kontrol tanpa bundling."))

    if loyal:
        r = sg.loc[loyal]
        rev = r.Nilai * upl
        items.append(dict(
            nama=f"Program loyalitas segmen {loyal}", rev=rev, cost=rev * promo,
            urg=1 + 4 * min(1.0, r.pv / 0.5), mudah=3.0,
            alasan=f"Segmen ini menyumbang {_pct(r.pv)} nilai belanja; kehilangannya paling merugikan.",
            asumsi=f"Belanja {angka(r.Jumlah)} pelanggan loyal naik {A['as_loyal']:.1f}%; biaya program {A['as_promo']:.0f}% dari tambahan penjualan.",
            rumus="Total belanja segmen loyal x kenaikan belanja",
            kpi="Retensi segmen loyal; rata-rata belanja per pelanggan loyal",
            ukur="Bandingkan anggota program vs non-anggota (kontrol) tiap bulan."))

    if risk:
        r = sg.loc[risk]
        react_n = r.Jumlah * react
        rev = react_n * r.Monetary
        items.append(dict(
            nama=f"Win-back segmen {risk}", rev=rev, cost=react_n * voucher,
            urg=1 + 4 * min(1.0, r.pc / 0.3), mudah=3.0,
            alasan=f"{_pct(r.pc)} pelanggan berisiko tidak kembali (rata-rata {r.Recency:.0f} hari sejak transaksi terakhir).",
            asumsi=f"{A['as_react']:.1f}% dari {angka(r.Jumlah)} pelanggan kembali aktif dengan belanja setara rata-rata historis "
                   f"({rupiah(r.Monetary)}); voucher {rupiah(voucher)} per pelanggan yang kembali.",
            rumus="Pelanggan berisiko x tingkat reaktivasi x rata-rata Monetary",
            kpi="Tingkat reaktivasi; penurunan porsi pelanggan berisiko churn",
            ukur="Kirim kampanye ke 50% pelanggan berisiko (uji), 50% lainnya kontrol; hitung yang bertransaksi dalam 30-60 hari."))

    if mid:
        sub = sg.loc[mid]
        tot_freq = (sub.Frequency * sub.Jumlah).sum()
        avg_val = sub.Nilai.sum() / max(tot_freq, 1)
        extra_trx = tot_freq * nfreq
        rev = extra_trx * avg_val
        items.append(dict(
            nama="Pertumbuhan pelanggan baru & potensial", rev=rev, cost=extra_trx * voucher,
            urg=1 + 4 * min(1.0, sub.pc.sum() / 0.8), mudah=4.0,
            alasan=f"{_pct(sub.pc.sum())} pelanggan berada di segmen ini; peluang menaikkan frekuensi paling luas.",
            asumsi=f"Frekuensi transaksi segmen naik {A['as_newfreq']:.1f}% (±{angka(extra_trx)} transaksi tambahan) bernilai "
                   f"{rupiah(avg_val)}/transaksi; voucher {rupiah(voucher)} per transaksi tambahan.",
            rumus="Transaksi tambahan x rata-rata nilai transaksi segmen",
            kpi="Repeat purchase rate; frekuensi rata-rata segmen",
            ukur="Voucher pembelian kedua pada separuh pelanggan baru (uji) vs separuh lain (kontrol)."))

    if len(monthly) >= 3:
        low = monthly[monthly < monthly.median()]
        rev = low.sum() * season
        gap = 1 - monthly.min() / monthly.max() if monthly.max() > 0 else 0
        items.append(dict(
            nama="Kalender promo musiman (bulan lemah)", rev=rev, cost=rev * promo,
            urg=1 + 4 * min(1.0, gap / 0.5), mudah=4.0,
            alasan=f"Penjualan bulan terendah {_pct(gap)} di bawah bulan puncak.",
            asumsi=f"Penjualan pada bulan di bawah median ({', '.join(m.strftime('%b %Y') for m in low.index)}) naik "
                   f"{A['as_season']:.1f}%; biaya promo {A['as_promo']:.0f}% dari tambahan penjualan.",
            rumus="Penjualan bulan lemah x kenaikan penjualan",
            kpi="Penjualan bulan lemah vs periode sebelumnya",
            ukur="Bandingkan penjualan bulan promo dengan rata-rata bulan lemah sebelumnya."))
    return items, extra_items, exposed, months, sg


def score_items(items):
    if not items:
        return []
    maxnet = max(max(i["rev"] - i["cost"], 0) for i in items) or 1
    maxcost = max(i["cost"] for i in items) or 1
    for i in items:
        i["net"] = i["rev"] - i["cost"]
        i["s_manfaat"] = 1 + 4 * max(i["net"], 0) / maxnet
        i["s_biaya"] = 5 - 4 * (i["cost"] / maxcost)
        i["skor"] = 0.3 * i["urg"] + 0.3 * i["s_manfaat"] + 0.2 * i["s_biaya"] + 0.2 * i["mudah"]
    return sorted(items, key=lambda i: i["skor"], reverse=True)


# ---------------------------------------------------------------------
# Pembuktian dengan data (e)
# ---------------------------------------------------------------------
def validate_rules(d, rules, basket, top=5):
    """Uji aturan Apriori: chi-square pada seluruh data + validasi antar-periode (awal vs akhir)."""
    from scipy.stats import chi2_contingency
    first = d.groupby("Transaction_ID")["Transaction_Date"].min().reindex(basket.index)
    cut = first.median()
    parts = {"all": basket, "awal": basket[first <= cut], "akhir": basket[first > cut]}

    def stats(b, Ai, Bi):
        n = len(b)
        if n == 0:
            return n, 0, 0, 0, np.nan, np.nan
        a, bb = b[Ai].all(axis=1), b[Bi].all(axis=1)
        nA, nB, nAB = int(a.sum()), int(bb.sum()), int((a & bb).sum())
        conf = nAB / nA if nA else np.nan
        lift = conf / (nB / n) if nA and nB else np.nan
        return n, nA, nB, nAB, conf, lift

    rows = []
    for _, r in rules.head(top).iterrows():
        Ai, Bi = r["antecedents"].split(", "), r["consequents"].split(", ")
        if any(x not in basket.columns for x in Ai + Bi):
            continue
        n, nA, nB, nAB, conf, lift = stats(parts["all"], Ai, Bi)
        try:
            p = float(chi2_contingency([[nAB, nA - nAB], [nB - nAB, n - nA - nB + nAB]])[1])
        except Exception:
            p = np.nan
        lt = stats(parts["awal"], Ai, Bi)[5]
        le = stats(parts["akhir"], Ai, Bi)[5]
        sig = (not np.isnan(p)) and p < 0.05
        if lift > 1 and sig and lt > 1 and le > 1:
            status = "Terbukti signifikan & konsisten"
        elif lift > 1 and lt > 1 and le > 1:
            status = "Konsisten, belum signifikan"
        elif lift > 1:
            status = "Belum konsisten antar periode"
        else:
            status = "Belum terbukti"
        rows.append({"Aturan": f"{r['antecedents']} → {r['consequents']}", "Transaksi A": nA, "Transaksi A+B": nAB,
                     "Confidence": conf, "Lift": lift, "p-value": p, "Lift periode awal": lt,
                     "Lift periode akhir": le, "Status": status})
    return pd.DataFrame(rows), cut


def segment_tests(seg):
    from scipy.stats import kruskal
    groups = [g for _, g in seg.groupby("Segmen")]
    out = []
    if len(groups) < 2:
        return out
    for col in ["Recency", "Frequency", "Monetary"]:
        try:
            h, p = kruskal(*[g[col].values for g in groups])
            out.append((col, float(h), float(p)))
        except Exception:
            pass
    return out


def build_summary(d, seg, rules, sg, pareto):
    total = d["Line_Total"].sum()
    n_trx = d["Transaction_ID"].nunique()
    n_cust = d["Customer_ID"].nunique()
    basket_sz = d.groupby("Transaction_ID")["Product_ID"].nunique().mean()
    monthly = d.groupby("Month")["Line_Total"].sum()
    m_best, m_low = monthly.idxmax(), monthly.idxmin()
    cat = d.groupby("Category")["Line_Total"].sum().sort_values(ascending=False)
    prod = d.groupby("Product_Name")["Line_Total"].sum().sort_values(ascending=False)
    reg = (d.groupby("Region")["Line_Total"].sum().drop("Tidak Diketahui", errors="ignore")
           .sort_values(ascending=False))
    pay = d["Payment_Method"].value_counts().drop("Tidak Diketahui", errors="ignore")
    ctype = (d.groupby("Customer_Type")["Line_Total"].sum().drop("Tidak Diketahui", errors="ignore")
             .sort_values(ascending=False))
    kes = [
        f"**Kinerja penjualan.** Total penjualan {rupiah(total)} dari {angka(n_trx)} transaksi dan "
        f"{angka(n_cust)} pelanggan. Nilai rata-rata per transaksi {rupiah(total / n_trx)} dengan rata-rata "
        f"{basket_sz:.2f} produk unik per keranjang"
        + (" (keranjang kecil, ruang cross-selling masih besar)." if basket_sz < 2 else "."),
        f"**Kapan penjualan meningkat?** Puncak pada {m_best:%B %Y} ({rupiah(monthly.max())}), "
        f"terendah pada {m_low:%B %Y} ({rupiah(monthly.min())}).",
        f"**Produk & kategori.** Kategori {cat.index[0]} terbesar ({_pct(cat.iloc[0] / total)}). "
        f"Lima produk teratas menyumbang {_pct(prod.head(5).sum() / total)} penjualan, dipimpin {prod.index[0]}.",
    ]
    if len(reg) >= 2:
        kes.append(f"**Wilayah.** {reg.index[0]} terbesar ({_pct(reg.iloc[0] / total)}), "
                   f"{reg.index[-1]} terkecil ({_pct(reg.iloc[-1] / total)}).")
    if len(ctype) >= 1:
        kes.append("**Tipe pelanggan.** " + ", ".join(f"{k} {_pct(v / total)}" for k, v in ctype.items()) + " dari total penjualan.")
    if len(pay) >= 1:
        kes.append(f"**Pembayaran.** Metode terbanyak: {pay.index[0]} ({_pct(pay.iloc[0] / pay.sum())} transaksi).")
    if rules is not None and not rules.empty:
        pairs = [f"{r['antecedents']} → {r['consequents']} (confidence {_pct(r['confidence'])}, lift {r['lift']:.2f})"
                 for _, r in rules.head(3).iterrows()]
        kes.append("**Produk yang sering dibeli bersama (Apriori).** " + "; ".join(pairs) + ".")
    else:
        kes.append("**Produk yang sering dibeli bersama (Apriori).** Belum ada aturan kuat; turunkan support/confidence.")
    kes.append("**Segmen paling bernilai (K-Means).** "
               + "; ".join(f"{s} ({_pct(r.pc)} pelanggan, {_pct(r.pv)} nilai belanja)" for s, r in sg.iterrows())
               + f". 20% pelanggan teratas menyumbang {_pct(pareto)} nilai belanja.")
    return kes, dict(total=total, n_trx=n_trx, n_cust=n_cust, basket=basket_sz)


def page_kesimpulan():
    page_header("📝", "Kesimpulan & Rekomendasi Keputusan Bisnis",
                "Semua dalam satu halaman: kesimpulan, lalu rekomendasi a-f beserta pembuktian dari data.")
    if not need_clean():
        return
    d = st.session_state["clean"]
    n_raw = len(st.session_state["raw"])
    seg = st.session_state.get("rfm_seg")
    rules = st.session_state.get("rules")
    notes = []

    # bila menu 4 / 5 belum dijalankan, hitung otomatis dengan cara yang sama
    if rules is None or rules.empty:
        try:
            s, c = auto_apriori_params(d)
            _, rules = get_rules(d, s, c, cap=N_RULES)
            notes.append(f"Apriori dihitung otomatis (support {s:.4f}, confidence {c:.2f}) karena menu 4 belum dibuka.")
        except Exception:
            rules = None
    if seg is None:
        seg, k_auto = auto_segment(st.session_state["rfm"])
        notes.append(f"Segmentasi memakai k = {k_auto} (titik elbow otomatis) karena menu 5 belum dibuka.")
    for n in notes:
        st.info(n)

    rules = dedup_rules(rules)
    basket = make_basket(d)
    n_trx = basket.shape[0]
    A = {k: asm(k) for k in ASSUMP}
    items, extra_items, exposed, months, sg = estimate_impact(d, seg, rules, basket, A)
    scored = score_items([dict(i) for i in items])
    loyal, risk, mid = seg_groups(sg)
    cs = seg.sort_values("Monetary", ascending=False)
    top_n = max(1, int(np.ceil(0.2 * len(cs))))
    pareto = cs["Monetary"].head(top_n).sum() / cs["Monetary"].sum()
    kes, kp = build_summary(d, seg, rules, sg, pareto)
    has_rules = rules is not None and not rules.empty
    report = []   # (judul, isi teks) untuk unduhan

    # tanpa tab: semua bagian ditampilkan berurutan dalam satu halaman (masing-masing satu kali)
    tabs = [st.container() for _ in range(7)]

    # ================= KESIMPULAN =================
    with tabs[0]:
        section("📌 Kesimpulan", f"Ringkasan hasil analisis (periode data ± {months:.1f} bulan).")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total penjualan", rupiah(kp["total"]))
        c2.metric("Jumlah transaksi", angka(kp["n_trx"]))
        c3.metric("Pelanggan unik", angka(kp["n_cust"]))
        c4.metric("Rata-rata basket_size", f"{kp['basket']:.2f}")
        section("Kesimpulan Analisis")
        for i, k in enumerate(kes, 1):
            st.markdown(f"{i}. {k}")
        st.text_area("Tambahkan kesimpulan Anda sendiri:", height=100, key="kes_tambahan")
        section("Catatan & keterbatasan")
        st.markdown(
            "- Angka estimasi dampak bergantung pada **asumsi** yang dapat diubah di tab *d. Estimasi Dampak*.\n"
            "- Aturan Apriori menunjukkan keterkaitan, **bukan sebab-akibat**; karena itu diuji statistik dan A/B test.\n"
            "- Label segmen K-Means dibuat otomatis dari profil RFM dan bergantung pada jumlah klaster (k).\n"
            "- Kesimpulan hanya mencerminkan periode dan kualitas data yang diunggah."
        )
        report.append(("KESIMPULAN", "\n".join(f"{i}. {k.replace('**', '')}" for i, k in enumerate(kes, 1))))

    # ================= a. PENEMPATAN PRODUK =================
    with tabs[1]:
        section("a. Strategi Penempatan Produk",
                "Ringkasan: produk yang sebaiknya ditempatkan berdekatan atau dijual sebagai bundling (hasil Apriori).")
        if has_rules:
            promo = A["as_promo"] / 100
            up = d.groupby("Product_Name")["Unit_Price"].mean()
            rows = []
            for i, (_, r) in enumerate(rules.head(5).iterrows(), 1):
                items_ab = r["antecedents"].split(", ") + r["consequents"].split(", ")
                normal = sum(up.get(x, 0.0) for x in items_ab)
                rows.append({
                    "No": i, "Pasangan produk": f"{r['antecedents']} + {r['consequents']}",
                    "Support": f"{r['support']:.2%}", "Confidence": f"{r['confidence']:.1%}", "Lift": f"{r['lift']:.2f}",
                    "Transaksi": angka(round(r["support"] * n_trx)),
                    "Tindakan": "Bundling + tempatkan berdekatan" if r["lift"] >= 1.5 else "Tempatkan berdekatan + tawarkan di kasir",
                    "Harga normal": rupiah(normal), f"Harga paket (diskon {promo:.0%})": rupiah(normal * (1 - promo)),
                })
            tab_a = pd.DataFrame(rows)
            report.append(("a. STRATEGI PENEMPATAN PRODUK", tab_a.to_string(index=False)))
            r0 = rules.iloc[0]
            st.markdown(
                "Letakkan produk yang sering dibeli bersamaan secara berdekatan atau tawarkan sebagai paket bundling. "
                f"Pasangan terkuat: **{r0['antecedents']}** dan **{r0['consequents']}** "
                f"(confidence {r0['confidence']:.1%}, lift {r0['lift']:.2f})."
            )
        else:
            st.warning("Belum ada aturan Apriori yang dapat dijadikan dasar rekomendasi penempatan produk.")

    # ================= b. RETENSI PELANGGAN =================
    with tabs[2]:
        section("b. Strategi Retensi Pelanggan",
                "Ringkasan: cara mempertahankan pelanggan bernilai tinggi dan menangani pelanggan berisiko churn (hasil K-Means).")
        rows = []
        for s, r in sg.iterrows():
            base = s.rsplit(" ", 1)[0] if s[-1].isdigit() else s
            rows.append({"Segmen": s, "Pelanggan": angka(r.Jumlah), "% Pelanggan": _pct(r.pc), "% Nilai belanja": _pct(r.pv),
                         "Recency (hari)": f"{r.Recency:.0f}", "Frequency": f"{r.Frequency:.1f}",
                         "Rata-rata Monetary": rupiah(r.Monetary), "Tindakan pemasaran": STRATEGI.get(base, "-")})
        tab_b = pd.DataFrame(rows)
        report.append(("b. STRATEGI RETENSI PELANGGAN", tab_b.to_string(index=False)))
        if loyal:
            r = sg.loc[loyal]
            st.markdown(f"- **{loyal}** ({_pct(r.pc)} pelanggan, {_pct(r.pv)} nilai belanja): pertahankan lewat program poin/tier "
                        "dan layanan prioritas tanpa diskon berlebihan.")
        if risk:
            r = sg.loc[risk]
            st.markdown(f"- **{risk}** ({_pct(r.pc)} pelanggan, rata-rata {r.Recency:.0f} hari sejak transaksi terakhir): "
                        "jalankan kampanye win-back dengan voucher terbatas waktu dan pesan personal.")
            tgt = seg[seg["Segmen"] == risk].sort_values("Monetary", ascending=False)
            st.download_button("⬇️ Unduh daftar pelanggan berisiko untuk win-back (CSV)",
                               tgt[["Customer_ID", "Recency", "Frequency", "Monetary"]].to_csv(index=False).encode("utf-8"),
                               "daftar_winback.csv", "text/csv")
        if not loyal and not risk:
            st.info("Segmen loyal dan berisiko belum teridentifikasi.")

    # ================= c. PRIORITAS TINDAKAN =================
    with tabs[3]:
        section("c. Prioritas Tindakan",
                "Ringkasan: urutan rekomendasi berdasarkan urgensi, potensi manfaat, biaya, dan kemudahan (skala 1-5).")
        if scored:
            rows = [{"Peringkat": n, "Rekomendasi": i["nama"], "Urgensi": f"{i['urg']:.1f}",
                     "Potensi manfaat": f"{i['s_manfaat']:.1f}", "Biaya (5 = murah)": f"{i['s_biaya']:.1f}",
                     "Kemudahan": f"{i['mudah']:.1f}", "Skor akhir": f"{i['skor']:.2f}",
                     "Estimasi dampak bersih": rupiah(i["net"])} for n, i in enumerate(scored, 1)]
            tab_c = pd.DataFrame(rows)
            report.append(("c. PRIORITAS TINDAKAN", tab_c.to_string(index=False)))
            st.markdown("Urutan prioritas: " + " → ".join(f"**{n}. {i['nama']}**" for n, i in enumerate(scored, 1)) + ".")
            st.caption("Skor akhir = 30% urgensi + 30% potensi manfaat + 20% biaya + 20% kemudahan.")
        else:
            st.warning("Belum ada rekomendasi yang dapat dinilai.")

    # ================= d. ESTIMASI DAMPAK =================
    with tabs[4]:
        section("d. Estimasi Dampak Bisnis",
                f"Ringkasan: perkiraan tambahan penjualan per periode setara data (± {months:.1f} bulan).")
        if scored:
            rows = [{"Rekomendasi": i["nama"], "Pendapatan tambahan": rupiah(i["rev"]), "Biaya": rupiah(i["cost"]),
                     "Dampak bersih": rupiah(i["net"]),
                     "ROI": f"{(i['net'] / i['cost']):.1f}x" if i["cost"] > 0 else "-"} for i in scored]
            tot_rev = sum(i["rev"] for i in scored)
            tot_cost = sum(i["cost"] for i in scored)
            rows.append({"Rekomendasi": "TOTAL (kasar)", "Pendapatan tambahan": rupiah(tot_rev), "Biaya": rupiah(tot_cost),
                         "Dampak bersih": rupiah(tot_rev - tot_cost), "ROI": "-"})
            tab_d = pd.DataFrame(rows)
            tab_d2 = pd.DataFrame([{"Rekomendasi": i["nama"], "Asumsi": i["asumsi"], "Rumus": i["rumus"],
                                    "Indikator keberhasilan": i["kpi"], "Cara mengukur": i["ukur"]} for i in scored])
            report.append(("d. ESTIMASI DAMPAK BISNIS", tab_d.to_string(index=False) + "\n\n" + tab_d2.to_string(index=False)))
            best = max(scored, key=lambda i: i["net"])
            st.markdown(
                f"Total dampak bersih diperkirakan ± **{rupiah(tot_rev - tot_cost)}** "
                f"({_pct((tot_rev - tot_cost) / kp['total'])} dari penjualan periode data); kontribusi terbesar dari "
                f"**{best['nama']}**. Angka ini kasar karena program dapat saling tumpang tindih dan bergantung pada asumsi."
            )
        else:
            st.warning("Belum ada data yang cukup untuk estimasi dampak.")
        with st.expander("⚙️ Ubah asumsi estimasi (opsional)"):
            a1, a2, a3, a4 = st.columns(4)
            a1.number_input("Konversi bundling (%)", 0.0, 100.0, ASSUMP["as_conv"], 0.5, key="as_conv")
            a2.number_input("Kenaikan belanja loyal (%)", 0.0, 100.0, ASSUMP["as_loyal"], 0.5, key="as_loyal")
            a3.number_input("Reaktivasi berisiko (%)", 0.0, 100.0, ASSUMP["as_react"], 0.5, key="as_react")
            a4.number_input("Kenaikan frekuensi baru/potensial (%)", 0.0, 100.0, ASSUMP["as_newfreq"], 0.5, key="as_newfreq")
            b1, b2, b3 = st.columns(3)
            b1.number_input("Kenaikan penjualan bulan lemah (%)", 0.0, 100.0, ASSUMP["as_season"], 0.5, key="as_season")
            b2.number_input("Biaya promosi (% dari tambahan penjualan)", 0.0, 100.0, ASSUMP["as_promo"], 0.5, key="as_promo")
            b3.number_input("Nilai voucher (Rp)", 0, 1000000, ASSUMP["as_voucher"], 1000, key="as_voucher")

    # ================= e. RENCANA IMPLEMENTASI (DIBUKTIKAN) =================
    with tabs[5]:
        section("e. Rencana Implementasi",
                "Setiap langkah disertai bukti dari data; di bawahnya ada pengujian statistik untuk membuktikan dasar rekomendasi.")
        val, cut = (validate_rules(d, rules, basket) if has_rules else (pd.DataFrame(), None))
        removed, nocust = n_raw - len(d), int(d["Customer_ID"].isna().sum())
        monthly = d.groupby("Month")["Line_Total"].sum()
        low_m = ", ".join(m.strftime("%b %Y") for m in monthly[monthly < monthly.median()].index) if len(monthly) >= 3 else "-"
        top_status = val.iloc[0]["Status"] if not val.empty else "belum diuji"
        top_lift = f"{val.iloc[0]['Lift']:.2f}" if not val.empty else "-"
        top_p = _fmt_p(val.iloc[0]["p-value"]) if not val.empty else "-"

        plan = [("Jangka pendek (0-1 bulan)", "Standarisasi input kasir/POS dan pembersihan data",
                 "Kategori & metode bayar baku; ID produk/pelanggan wajib diisi",
                 f"{angka(removed)} baris ({_pct(removed / n_raw)}) dibuang saat pembersihan; {angka(nocust)} baris tanpa ID pelanggan",
                 "TI & Operasional")]
        if has_rules:
            plan.append(("Jangka pendek (0-1 bulan)", "Tata ulang rak/halaman produk: pasangkan produk hasil Apriori",
                         "3 pasangan produk teratas berdekatan",
                         f"Aturan teratas lift {top_lift}, p-value {top_p}: {top_status}", "Merchandising"))
        if risk:
            r = sg.loc[risk]
            plan.append(("Jangka pendek (0-1 bulan)", f"Kampanye win-back segmen {risk}", f"{angka(r.Jumlah)} pelanggan dihubungi",
                         f"{_pct(r.pc)} pelanggan, rata-rata {r.Recency:.0f} hari sejak transaksi terakhir", "CRM / Pemasaran"))
        if has_rules:
            plan.append(("Jangka menengah (1-3 bulan)", "Pilot bundling A/B selama 4-6 minggu",
                         "Kelompok uji vs kontrol; keputusan scale-up berdasar KPI",
                         f"±{angka(exposed / months)} transaksi/bulan terpapar peluang bundling", "Pemasaran & Operasional"))
        if loyal:
            r = sg.loc[loyal]
            plan.append(("Jangka menengah (1-3 bulan)", f"Program loyalitas segmen {loyal}", "Poin/tier aktif",
                         f"{angka(r.Jumlah)} pelanggan = {_pct(r.pv)} nilai belanja", "CRM / Pemasaran"))
        if mid:
            sub = sg.loc[mid]
            plan.append(("Jangka menengah (1-3 bulan)", "Voucher pembelian kedua bagi pelanggan baru/potensial", "Repeat purchase naik",
                         f"{angka(sub.Jumlah.sum())} pelanggan ({_pct(sub.pc.sum())}) di segmen {', '.join(mid)}", "CRM / Pemasaran"))
        plan.append(("Jangka menengah (3-6 bulan)", "Kalender promo musiman untuk bulan lemah", "Promo terjadwal sebelum bulan lemah",
                     f"Bulan di bawah median: {low_m}", "Pemasaran & Pembelian"))
        plan.append(("Jangka menengah (3-6 bulan)", "Evaluasi KPI bulanan, scale-up atau hentikan program", "Laporan KPI per bulan",
                     "Keputusan dari hasil uji vs kontrol (lihat tab f)", "Manajemen"))
        tab_e = pd.DataFrame(plan, columns=["Fase", "Tindakan", "Target / output", "Bukti dari data", "PIC (usulan)"])
        html_table(tab_e)
        report.append(("e. RENCANA IMPLEMENTASI", tab_e.to_string(index=False)))

        section("🔬 Pembuktian 1: pola produk teruji statistik dan antar-periode",
                f"Chi-square pada seluruh transaksi + perbandingan lift periode awal vs akhir (batas {cut:%d %b %Y})." if cut is not None else "")
        if not val.empty:
            v = val.copy()
            v["Confidence"] = v["Confidence"].map(lambda x: f"{x:.1%}")
            for c in ["Lift", "Lift periode awal", "Lift periode akhir"]:
                v[c] = v[c].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}")
            v["p-value"] = v["p-value"].map(_fmt_p)
            html_table(v)
            n_ok = int(val["Status"].str.startswith("Terbukti").sum())
            if n_ok:
                st.success(f"{n_ok} dari {len(val)} aturan teratas terbukti signifikan (p < 0.05) dan konsisten di kedua periode: layak dipilotkan.")
            else:
                st.warning("Belum ada aturan yang terbukti signifikan dan konsisten; jalankan pilot A/B kecil sebelum diperluas.")
            report.append(("PEMBUKTIAN 1 - POLA PRODUK", val.to_string(index=False)))
        else:
            st.info("Tidak ada aturan untuk diuji.")

        section("🔬 Pembuktian 2: segmen pelanggan memang berbeda",
                "Uji Kruskal-Wallis: apakah Recency, Frequency, Monetary berbeda antar segmen (p < 0.05 = berbeda nyata).")
        tests = segment_tests(seg)
        if tests:
            t = pd.DataFrame([{"Variabel": c, "Statistik H": f"{h:.1f}", "p-value": _fmt_p(p),
                               "Kesimpulan": "Berbeda nyata antar segmen" if p < 0.05 else "Tidak berbeda nyata"} for c, h, p in tests])
            html_table(t)
            st.markdown(f"20% pelanggan teratas menyumbang **{_pct(pareto)}** nilai belanja, sehingga fokus retensi pada segmen bernilai tinggi dibenarkan data.")
            report.append(("PEMBUKTIAN 2 - SEGMENTASI", t.to_string(index=False)))
        else:
            st.info("Segmen belum cukup untuk diuji.")

        section("🔬 Pembuktian 3: kelayakan beban kerja")
        if has_rules:
            st.markdown(f"- Bundling: ±**{angka(exposed / months)}** transaksi/bulan menjadi sasaran penawaran; cukup besar untuk uji A/B 4-6 minggu.")
        if risk:
            st.markdown(f"- Win-back: **{angka(sg.loc[risk].Jumlah)}** pelanggan; dapat dijalankan bertahap lewat pesan/WA/email.")
        st.markdown(f"- Data siap pakai: {_pct(1 - removed / n_raw)} baris lolos pembersihan; {angka(nocust)} baris tanpa ID pelanggan perlu perbaikan di sumber.")

    # ================= f. EVALUASI KPI =================
    with tabs[6]:
        section("f. Evaluasi: KPI Keberhasilan",
                "Ringkasan: indikator yang dipantau untuk menilai berhasil atau tidaknya rekomendasi.")
        tot_rev = sum(i["rev"] for i in items)
        kp_rows = [
            ("Penjualan rata-rata per bulan", "Total penjualan / jumlah bulan", rupiah(kp["total"] / months),
             rupiah((kp["total"] + tot_rev) / months), "Bulanan"),
            ("Rata-rata basket_size", "Produk unik per transaksi", f"{kp['basket']:.2f}",
             f"{kp['basket'] + extra_items / max(n_trx, 1):.2f}", "Mingguan"),
        ]
        if has_rules:
            conf = rules.iloc[0]["confidence"]
            kp_rows.append(("Attach rate pasangan produk teratas", "Transaksi A+B / transaksi A", _pct(conf),
                            _pct(conf + (1 - conf) * A["as_conv"] / 100), "Mingguan"))
        rep_rate = (seg["Frequency"] >= 2).mean()
        kp_rows.append(("Repeat purchase rate", "% pelanggan dengan ≥ 2 transaksi", _pct(rep_rate),
                        _pct(min(1.0, rep_rate + 0.05)) + " (usulan +5 poin)", "Bulanan"))
        if risk:
            kp_rows.append(("Porsi pelanggan berisiko churn", "Pelanggan segmen berisiko / total pelanggan", _pct(sg.loc[risk].pc),
                            _pct(sg.loc[risk].pc * (1 - A["as_react"] / 100)), "Bulanan"))
            kp_rows.append(("Tingkat reaktivasi", "Pelanggan berisiko yang bertransaksi lagi / yang dihubungi", "0%",
                            f"{A['as_react']:.1f}%", "Per kampanye"))
        if loyal:
            kp_rows.append(("Rata-rata belanja pelanggan loyal", "Monetary rata-rata segmen loyal", rupiah(sg.loc[loyal].Monetary),
                            rupiah(sg.loc[loyal].Monetary * (1 + A["as_loyal"] / 100)), "Bulanan"))
        kp_rows.append(("Rasio biaya promosi", "Biaya promosi / tambahan penjualan", "-", f"≤ {A['as_promo']:.0f}%", "Bulanan"))
        tab_f = pd.DataFrame(kp_rows, columns=["KPI", "Rumus", "Baseline (data saat ini)", "Target", "Frekuensi review"])
        report.append(("f. EVALUASI KPI", tab_f.to_string(index=False)))
        st.markdown("KPI yang dipantau: " + "; ".join(r[0] for r in kp_rows) + ".")
        st.markdown("Evaluasi dengan membandingkan kelompok uji dan kontrol. Perluas program (scale-up) bila target tercapai "
                    "pada 2 periode berturut-turut; revisi atau hentikan bila hasilnya tidak lebih baik dari kontrol.")

    txt = ["LAPORAN KESIMPULAN & REKOMENDASI KEPUTUSAN BISNIS", "=" * 50, ""]
    for title, body in report:
        txt += [title, "-" * len(title), body, ""]
    st.divider()
    st.download_button("⬇️ Unduh laporan lengkap kesimpulan & rekomendasi (TXT)", "\n".join(txt).encode("utf-8"),
                       "laporan_rekomendasi.txt", "text/plain")


# =====================================================================
# MAIN
# =====================================================================
def main():
    st.set_page_config(page_title="UTS Business Intelligence", page_icon="📊", layout="wide")
    apply_blue_theme()

    for k in ["raw", "produk", "pelanggan", "filename", "clean", "log", "rfm",
              "ref_date", "rules", "rfm_seg", "upload_sig", "goto", "auth", "user"]:
        st.session_state.setdefault(k, None)

    # ---- gerbang login: tampil lebih dulu sebelum menu utama ----
    if not st.session_state["auth"]:
        page_login()
        return

    # pindah menu otomatis (dipicu setelah upload berhasil)
    if st.session_state.get("goto"):
        st.session_state["menu"] = st.session_state["goto"]
        st.session_state["goto"] = None

    show_sidebar_profile()
    st.sidebar.divider()
    menu = st.sidebar.radio("Menu", MENU, key="menu")

    # setiap pindah menu, halaman harus tampil dari paling atas
    if st.session_state.get("_last_menu") != menu:
        st.session_state["_last_menu"] = menu
        with st.sidebar:
            scroll_top(menu)

    st.sidebar.divider()
    st.sidebar.write("Data mentah:", "✅" if st.session_state["raw"] is not None else "❌ belum upload")
    st.sidebar.write("Data bersih:", "✅" if st.session_state["clean"] is not None else "❌ belum diproses")

    st.sidebar.divider()
    st.sidebar.write(f"👤 Masuk sebagai: **{st.session_state['user']}**")
    if st.sidebar.button("🚪 Logout"):
        st.session_state["auth"] = False
        st.session_state["user"] = None
        st.rerun()

    if menu == MENU[0]:
        page_upload()
    elif menu == MENU[1]:
        page_eda()
    elif menu == MENU[2]:
        page_preprocessing()
    elif menu == MENU[3]:
        page_apriori()
    elif menu == MENU[4]:
        page_kmeans()
    elif menu == MENU[5]:
        page_dashboard()
    else:
        page_kesimpulan()


if __name__ == "__main__":
    main()