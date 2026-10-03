import html
import sqlite3
from datetime import timedelta
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
DB = BASE / 'ecommerce_hackathon.db'
MODELS = BASE / 'models'; OUT = BASE / 'outputs'; PLOTS = BASE / 'plots'

st.set_page_config(page_title='E-Commerce Customer Intelligence', page_icon='🛍️', layout='wide',
                   initial_sidebar_state='collapsed')

# ============================================================== design tokens
FONT = 'Inter'
TEXT, SUB, LINE = '#1A1F36', '#6B7190', '#E6E8F0'
CANVAS, SURF2 = '#F1F3F9', '#F6F7FB'
NIGHT = '#141832'
YELLOW = '#F2C811'
BLUE, NAVY, ORANGE, PURPLE, PINK, VIOLET = '#118DFF', '#12239E', '#E66C37', '#6B007B', '#E044A7', '#744EC2'
GOOD, BAD, MID = '#16A34A', '#E5484D', '#E2A400'
PALETTE = ['#118DFF', '#12239E', '#E66C37', '#744EC2', '#E044A7', '#197278', '#D9B300', '#D64550', '#1AAB40', '#6B007B']

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp {{ font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }}
h1, h2, h3, h4, p, label, input, textarea, button {{ font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important; }}
.stApp {{ background: {CANVAS}; }}
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"] {{ display: none; }}
.block-container {{ padding: 0 1.8rem 3rem; max-width: 1480px; }}

@keyframes rise {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: none; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; transition: none !important; }} }}

/* ---------------- app bar with navigation ---------------- */
.st-key-appbar {{
    position: sticky; top: 0; z-index: 100; margin: 0 -1.8rem; padding: .55rem 1.8rem;
    background: linear-gradient(100deg, {NIGHT} 0%, #1E2350 55%, #26306B 100%);
    border-bottom: 3px solid {YELLOW}; box-shadow: 0 6px 20px rgba(20,24,50,.18);
}}
.brand {{ display: flex; align-items: center; gap: .7rem; color: #fff; }}
.logo {{
    width: 34px; height: 34px; flex: none; display: flex; align-items: flex-end; justify-content: center; gap: 3px;
    padding: 7px 6px; background: {YELLOW}; border-radius: 9px; box-shadow: 0 4px 14px rgba(242,200,17,.4);
}}
.logo span {{ display: block; width: 5px; background: {NIGHT}; border-radius: 1px; }}
.brand-name {{ font-weight: 700; font-size: 1rem; line-height: 1.15; }}
.brand-sub {{ font-size: .74rem; color: rgba(255,255,255,.6); }}
.st-key-nav [role="radiogroup"] {{
    gap: 4px; flex-wrap: nowrap; justify-content: center; background: rgba(255,255,255,.07);
    padding: 4px; border-radius: 12px; border: 1px solid rgba(255,255,255,.1); width: fit-content; margin: 0 auto;
}}
.st-key-nav label {{ margin: 0; padding: .45rem 1rem; border-radius: 9px; cursor: pointer; transition: background .15s; }}
.st-key-nav label > div:first-child {{ display: none; }}
.st-key-nav label p {{ color: rgba(255,255,255,.72); font-size: .88rem; font-weight: 500; white-space: nowrap; }}
.st-key-nav label:hover {{ background: rgba(255,255,255,.08); }}
.st-key-nav label:has(input:checked) {{ background: #fff; }}
.st-key-nav label:has(input:checked) p {{ color: {NIGHT}; font-weight: 700; }}
.bar-chips {{ display: flex; justify-content: flex-end; gap: .5rem; }}
.chip {{
    font-size: .74rem; padding: .28rem .7rem; border-radius: 999px; white-space: nowrap;
    background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.16); color: #E8EAF6;
}}
.chip.live::before {{
    content: ''; display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    background: #3DDC84; margin-right: .4rem; vertical-align: 1px; box-shadow: 0 0 0 3px rgba(61,220,132,.25);
}}

/* ---------------- page heading ---------------- */
.page-title {{ color: {TEXT}; font-size: 1.7rem; font-weight: 800; letter-spacing: -.02em; margin-top: 1.1rem; }}
.page-sub {{ color: {SUB}; font-size: .92rem; margin: .15rem 0 .4rem; }}

/* ---------------- tiles: any container keyed vis_* ---------------- */
[class*="st-key-vis_"] {{
    background: #fff; border: 1px solid {LINE}; border-radius: 14px; padding: 1rem 1.15rem .8rem;
    box-shadow: 0 1px 2px rgba(16,24,40,.04), 0 6px 18px rgba(16,24,40,.05);
}}
.vis-title {{ color: {TEXT}; font-size: 1rem; font-weight: 700; display: flex; align-items: center; gap: .5rem; }}
.vis-title::before {{ content: ''; width: 4px; height: 16px; border-radius: 2px; background: var(--a, {BLUE}); }}
.vis-sub {{ color: {SUB}; font-size: .8rem; margin: .1rem 0 .3rem; }}
.hint {{ color: {SUB}; font-size: .78rem; }}

/* ---------------- segmented controls: radios keyed seg_* ---------------- */
[class*="st-key-seg_"] [role="radiogroup"] {{
    display: inline-flex; flex-wrap: nowrap; gap: 2px; background: {SURF2}; padding: 3px;
    border-radius: 9px; border: 1px solid {LINE};
}}
[class*="st-key-seg_"] label {{ margin: 0; padding: .28rem .75rem; border-radius: 7px; cursor: pointer; }}
[class*="st-key-seg_"] label > div:first-child {{ display: none; }}
[class*="st-key-seg_"] label p {{ font-size: .8rem; color: {SUB}; white-space: nowrap; }}
[class*="st-key-seg_"] label:has(input:checked) {{ background: #fff; box-shadow: 0 1px 3px rgba(16,24,40,.14); }}
[class*="st-key-seg_"] label:has(input:checked) p {{ color: {TEXT}; font-weight: 700; }}
[class*="st-key-seg_"] {{ display: flex; justify-content: flex-end; }}

/* ---------------- KPI cards ---------------- */
.kpi {{
    background: #fff; border: 1px solid {LINE}; border-radius: 14px; padding: 1rem 1.05rem .75rem;
    box-shadow: 0 1px 2px rgba(16,24,40,.04), 0 6px 18px rgba(16,24,40,.05);
    position: relative; overflow: hidden; height: 100%; animation: rise .5s ease both; animation-delay: var(--d, 0s);
}}
.kpi::after {{
    content: ''; position: absolute; right: -40px; top: -40px; width: 120px; height: 120px; border-radius: 50%;
    background: radial-gradient(circle, color-mix(in srgb, var(--c) 16%, transparent), transparent 70%);
}}
.kpi-top {{ display: flex; align-items: center; justify-content: space-between; position: relative; z-index: 1; }}
.kpi-label {{ color: {SUB}; font-size: .82rem; font-weight: 600; display: flex; align-items: center; gap: .55rem; }}
.kpi-icon {{
    width: 32px; height: 32px; border-radius: 10px; display: grid; place-items: center; font-size: 1rem;
    background: color-mix(in srgb, var(--c) 14%, white);
}}
.kpi-value {{ color: {TEXT}; font-size: 1.8rem; font-weight: 800; margin: .5rem 0 0; letter-spacing: -.03em;
              font-variant-numeric: tabular-nums; }}
.kpi-note {{ color: {SUB}; font-size: .75rem; }}
.kpi svg {{ display: block; width: 100%; height: 40px; margin-top: .45rem; }}
.delta {{ font-size: .74rem; font-weight: 700; padding: .2rem .55rem; border-radius: 999px; white-space: nowrap; }}
.delta.up {{ background: #E6F6EC; color: #12803B; }}
.delta.down {{ background: #FDECEC; color: #C2363B; }}

/* ---------------- insights ---------------- */
.narr {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: .75rem; }}
.narr-item {{ background: {SURF2}; border: 1px solid #EDEFF5; border-radius: 10px; padding: .75rem .9rem; }}
.narr-k {{ color: {SUB}; font-size: .76rem; font-weight: 600; }}
.narr-v {{ color: {TEXT}; font-size: 1.02rem; font-weight: 700; margin-top: .15rem; }}
.narr-s {{ color: {SUB}; font-size: .78rem; }}

/* ---------------- cross-filter bar ---------------- */
.xf {{ display: flex; flex-wrap: wrap; gap: .45rem; align-items: center; color: {SUB}; font-size: .82rem; }}
.xf-tag {{ background: #E8F3FF; color: #0B5CAD; border: 1px solid #CFE6FF; border-radius: 999px; padding: .2rem .65rem; font-weight: 600; }}

/* ---------------- HTML tables ---------------- */
.tbl {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
.tbl th {{ text-align: left; color: {SUB}; font-weight: 600; font-size: .76rem; padding: .55rem .6rem; border-bottom: 1px solid {LINE}; }}
.tbl td {{ padding: .6rem .6rem; border-bottom: 1px solid #F0F1F6; color: {TEXT}; vertical-align: middle; }}
.tbl tr:last-child td {{ border-bottom: 0; }}
.tbl tr:hover td {{ background: {SURF2}; }}
.tbl .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.rank {{ display: inline-grid; place-items: center; width: 26px; height: 26px; border-radius: 8px; font-weight: 700; font-size: .78rem; background: {SURF2}; color: {SUB}; }}
.rank.r1 {{ background: #FFF4C2; color: #8A6A00; }} .rank.r2 {{ background: #EEF0F4; color: #555C70; }} .rank.r3 {{ background: #FBE6D8; color: #9A4A1C; }}
.who {{ display: flex; align-items: center; gap: .6rem; font-weight: 600; }}
.avatar {{ width: 30px; height: 30px; border-radius: 50%; display: grid; place-items: center; color: #fff; font-size: .74rem; font-weight: 700; flex: none; }}
.bar {{ display: flex; align-items: center; gap: .6rem; min-width: 180px; }}
.bar-track {{ flex: 1; height: 8px; background: #EEF0F6; border-radius: 999px; overflow: hidden; }}
.bar-fill {{ height: 100%; border-radius: 999px; background: linear-gradient(90deg, #7CC0FF, {BLUE}); }}
.bar-val {{ font-variant-numeric: tabular-nums; font-weight: 600; min-width: 96px; text-align: right; }}
.badge {{ padding: .2rem .6rem; border-radius: 999px; font-weight: 600; font-size: .78rem; }}

/* ---------------- inputs & buttons ---------------- */
[data-baseweb="select"] > div, [data-baseweb="input"], .stTextArea textarea {{ border-radius: 9px !important; }}
[data-baseweb="tag"] {{ background: {BLUE} !important; border-radius: 6px !important; }}
.stButton > button, .stDownloadButton > button {{
    border-radius: 9px; font-weight: 600; border: 1px solid {LINE}; background: #fff; color: {TEXT};
    transition: border-color .15s, color .15s;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color: {BLUE}; color: {BLUE}; }}
.stButton > button[kind="primary"] {{ background: {BLUE}; color: #fff; border: 0; box-shadow: 0 4px 12px rgba(17,141,255,.3); }}
.stButton > button[kind="primary"]:hover {{ background: #0B6FCC; color: #fff; }}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {{ outline: 2px solid {BLUE}; outline-offset: 2px; }}
[data-testid="stTabs"] [role="tab"] p {{ font-weight: 600; }}

/* ---------------- gauge & result blocks ---------------- */
.gauge {{ position: relative; width: 280px; height: 150px; margin: .6rem auto 0; overflow: hidden; }}
.gauge-arc {{ position: relative; width: 280px; height: 280px; border-radius: 50%;
              background: conic-gradient(from -90deg, {GOOD} 0deg, {MID} 90deg, {BAD} 180deg, transparent 180deg); }}
.gauge-arc::after {{ content: ''; position: absolute; inset: 38px; border-radius: 50%; background: #fff; }}
.gauge-needle {{ position: absolute; left: 138.5px; bottom: 10px; width: 3px; height: 108px; background: {TEXT};
                 border-radius: 3px; transform-origin: 50% 100%; transition: transform .6s cubic-bezier(.2,.8,.2,1); }}
.gauge-hub {{ position: absolute; left: 131px; bottom: 2px; width: 18px; height: 18px; border-radius: 50%; background: {TEXT}; border: 3px solid #fff; }}
.gauge-scale {{ display: flex; justify-content: space-between; width: 280px; margin: .25rem auto 0; color: {SUB}; font-size: .74rem; }}
.big {{ text-align: center; font-size: 2.6rem; font-weight: 800; letter-spacing: -.03em; margin-top: .3rem; line-height: 1.1; }}
.status {{ text-align: center; font-weight: 700; font-size: .92rem; padding: .55rem; border-radius: 10px; margin: .6rem 0 .3rem; }}
.mrow {{ display: grid; grid-template-columns: 150px 1fr 54px; gap: .7rem; align-items: center; margin: .45rem 0; font-size: .84rem; color: {TEXT}; }}
.mtrack {{ position: relative; height: 10px; background: #EEF0F6; border-radius: 999px; overflow: visible; }}
.mfill {{ height: 100%; border-radius: 999px; }}
.mmark {{ position: absolute; left: 50%; top: -4px; bottom: -4px; width: 2px; background: {TEXT}; opacity: .35; }}
.mval {{ text-align: right; font-weight: 700; font-variant-numeric: tabular-nums; }}
.ring {{ width: 150px; height: 150px; border-radius: 50%; margin: .4rem auto .2rem; display: grid; place-items: center; }}
.ring-in {{ width: 116px; height: 116px; border-radius: 50%; background: #fff; display: grid; place-items: center; text-align: center; }}
.ring-emoji {{ font-size: 2.2rem; line-height: 1; }}
.ring-pct {{ font-weight: 800; color: {TEXT}; font-size: 1.05rem; margin-top: .2rem; }}
.callout {{ border-radius: 10px; padding: .75rem .9rem; font-size: .88rem; line-height: 1.5; margin-top: .4rem; }}
.empty {{ color: {SUB}; text-align: center; padding: 2.6rem 1rem; font-size: .9rem; }}
.empty-icon {{ font-size: 2rem; margin-bottom: .4rem; }}

@media (max-width: 1000px) {{ .narr {{ grid-template-columns: 1fr 1fr; }} .bar-chips {{ display: none; }} }}
</style>
""", unsafe_allow_html=True)


# ============================================================== data + models
@st.cache_data
def sql_df(query):
    con = sqlite3.connect(DB); df = pd.read_sql_query(query, con); con.close(); return df

@st.cache_resource
def load_models():
    return {n: joblib.load(MODELS / f'{n}.joblib') for n in ['logistic_regression', 'random_forest', 'neural_network', 'sentiment_model']}

models = load_models()

@st.cache_data
def valid_orders():
    # Same validity rules and net-revenue formula as the original queries, one row per valid order line
    df = sql_df("""
        SELECT o.order_id, o.customer_id, date(o.order_date) order_date, o.returned,
               o.quantity*o.unit_price*(1-o.discount)*(1-o.returned) net_revenue,
               p.category, c.customer_name, c.city
        FROM orders o
        LEFT JOIN products p ON o.product_id=p.product_id
        LEFT JOIN customers c ON o.customer_id=c.customer_id
        WHERE date(o.order_date) IS NOT NULL AND o.quantity>0 AND o.unit_price>=0""")
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['category'] = df['category'].fillna('Uncategorised')
    df['city'] = df['city'].fillna('Unknown')
    df['customer_name'] = df['customer_name'].fillna('Unknown customer')
    return df


# ============================================================== helpers
def esc(x):
    return html.escape(str(x))

def compact(v):
    for div, suf in [(1e9, 'bn'), (1e6, 'M'), (1e3, 'K')]:
        if abs(v) >= div:
            return f'{v / div:.2f}{suf}'
    return f'{v:,.0f}'

def sparkline(values, color, uid):
    vals = [float(v) for v in values if pd.notna(v)]
    if len(vals) < 2:
        return ''
    w, h, pad = 200, 40, 4
    lo, hi = min(vals), max(vals); span = (hi - lo) or 1
    pts = [(i * w / (len(vals) - 1), pad + (h - 2 * pad) * (1 - (v - lo) / span)) for i, v in enumerate(vals)]
    line = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
    lx, ly = pts[-1]
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none"><defs>'
            f'<linearGradient id="g{uid}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{color}" stop-opacity=".3"/>'
            f'<stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
            f'<polygon points="0,{h} {line} {w},{h}" fill="url(#g{uid})"/>'
            f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2.2" vector-effect="non-scaling-stroke"/>'
            f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="3.2" fill="{color}"/></svg>')

def kpi(col, uid, icon, color, label, value, note='', series=None, delta=None, lower_is_better=False, delay=0.0):
    d_html = ''
    if delta is not None and pd.notna(delta):
        good = (delta < 0) if lower_is_better else (delta >= 0)
        d_html = f'<span class="delta {"up" if good else "down"}">{"▲" if delta >= 0 else "▼"} {abs(delta):.1%}</span>'
    spark = sparkline(series, color, uid) if series is not None else ''
    note_html = f'<div class="kpi-note">{note}</div>' if note else ''
    col.markdown(f'<div class="kpi" style="--c:{color};--d:{delay}s"><div class="kpi-top"><div class="kpi-label">'
                 f'<span class="kpi-icon">{icon}</span>{label}</div>{d_html}</div>'
                 f'<div class="kpi-value">{value}</div>{note_html}{spark}</div>', unsafe_allow_html=True)

def title(text, sub='', accent=BLUE, target=st):
    sub_html = f'<div class="vis-sub">{sub}</div>' if sub else ''
    target.markdown(f'<div class="vis-title" style="--a:{accent}">{text}</div>{sub_html}', unsafe_allow_html=True)

def styled(chart):
    return (chart.properties(background='transparent')
            .configure_view(strokeWidth=0)
            .configure_axis(labelFont=FONT, labelColor=SUB, labelFontSize=11, titleFont=FONT, titleColor=SUB,
                            titleFontSize=11, titleFontWeight='normal', gridColor='#EEF0F5', domain=False,
                            ticks=False, labelPadding=6)
            .configure_axisX(grid=False, domain=True, domainColor='#DDE0E9')
            .configure_legend(labelFont=FONT, labelColor=TEXT, labelFontSize=11, symbolType='circle',
                              symbolSize=110, title=None, rowPadding=6)
            .configure_text(font=FONT))

def show(chart, **kw):
    return st.altair_chart(styled(chart), use_container_width=True, theme=None, **kw)

def picked(key, param, field):
    """Values clicked in a selectable Altair chart (cross-filtering)."""
    try:
        sel = st.session_state[key]['selection'][param]
        return [d[field] for d in sel if isinstance(d, dict) and field in d]
    except Exception:
        return []

def initials(name):
    parts = [p for p in str(name).split() if p]
    return (''.join(p[0] for p in parts[:2]) or '?').upper()


# ============================================================== app bar + navigation
PAGES = ['Dashboard', 'Churn Prediction', 'Sentiment Analysis']
NAV = {'Dashboard': '📊  Sales overview', 'Churn Prediction': '🔮  Churn prediction', 'Sentiment Analysis': '💬  Review sentiment'}

with st.container(key='appbar'):
    b1, b2, b3 = st.columns([1.3, 2.4, 1.3], vertical_alignment='center')
    b1.markdown('<div class="brand"><div class="logo"><span style="height:7px"></span><span style="height:12px"></span>'
                '<span style="height:19px"></span></div><div><div class="brand-name">Customer Intelligence</div>'
                '<div class="brand-sub">E-commerce analytics</div></div></div>', unsafe_allow_html=True)
    with b2:
        page = st.radio('Page', PAGES, format_func=NAV.get, horizontal=True, label_visibility='collapsed', key='nav')
    b3.markdown('<div class="bar-chips"><span class="chip live">Live · SQLite</span>'
                '<span class="chip">Models to 31 May 2026</span></div>', unsafe_allow_html=True)


# ============================================================== DASHBOARD
if page == 'Dashboard':
    df_all = valid_orders()
    total_customers = int(sql_df("SELECT COUNT(*) n FROM customers")['n'].iloc[0])
    dmin, dmax = df_all['order_date'].min().date(), df_all['order_date'].max().date()

    RANGES = ['All time', 'Last 30 days', 'Last 90 days', 'Last 6 months', 'Last 12 months', 'Custom range']
    for k, v in {'f_range': 'All time', 'f_cats': [], 'f_cities': []}.items():
        st.session_state.setdefault(k, v)

    def reset_filters():
        st.session_state.f_range = 'All time'; st.session_state.f_cats = []; st.session_state.f_cities = []
        for k in ('cat_chart', 'city_chart'):
            try:
                del st.session_state[k]
            except Exception:
                pass

    h1, h2 = st.columns([3, 1.2], vertical_alignment='bottom')
    h1.markdown(f'<div class="page-title">Sales overview</div><div class="page-sub">Net revenue after discounts and '
                f'returns · data from {dmin:%d %b %Y} to {dmax:%d %b %Y}</div>', unsafe_allow_html=True)

    # ----- filters
    with st.container(key='vis_filters'):
        f1, f2, f3, f4 = st.columns([1.3, 1.5, 2.2, 2.2], vertical_alignment='bottom')
        rng = f1.selectbox('📅 Period', RANGES, key='f_range')
        if rng == 'Custom range':
            period = f2.date_input('Dates', value=(dmin, dmax), min_value=dmin, max_value=dmax, format='DD/MM/YYYY')
            start, end = period if isinstance(period, (tuple, list)) and len(period) == 2 else (dmin, dmax)
        else:
            days_back = {'All time': None, 'Last 30 days': 30, 'Last 90 days': 90, 'Last 6 months': 182, 'Last 12 months': 365}[rng]
            start, end = (dmin if days_back is None else max(dmin, dmax - timedelta(days=days_back))), dmax
            f2.markdown(f'<div class="hint" style="padding-bottom:.6rem">{start:%d %b %Y} → {end:%d %b %Y}</div>',
                        unsafe_allow_html=True)
        cats = f3.multiselect('🏷️ Category', sorted(df_all['category'].unique()), key='f_cats', placeholder='All categories')
        cities = f4.multiselect('📍 City', sorted(df_all['city'].unique()), key='f_cities', placeholder='All cities')

    base = df_all[(df_all['order_date'].dt.date >= start) & (df_all['order_date'].dt.date <= end)]
    if cats:
        base = base[base['category'].isin(cats)]
    if cities:
        base = base[base['city'].isin(cities)]

    # ----- cross-filter from chart clicks
    pick_cat = picked('cat_chart', 'cat_pick', 'category')
    pick_city = picked('city_chart', 'city_pick', 'city')
    view_for_cat = base[base['city'].isin(pick_city)] if pick_city else base
    view_for_city = base[base['category'].isin(pick_cat)] if pick_cat else base
    df = view_for_cat[view_for_cat['category'].isin(pick_cat)] if pick_cat else view_for_cat

    with h2:
        a, b = st.columns(2)
        a.button('↺ Reset', on_click=reset_filters, use_container_width=True, help='Clear all filters and chart selections')
        b.download_button('⬇ Export', df.to_csv(index=False).encode('utf-8'), 'filtered_orders.csv', 'text/csv',
                          use_container_width=True, help='Download the orders behind the current view as CSV')

    tags = [f'<span class="xf-tag">Category: {esc(c)}</span>' for c in pick_cat] + \
           [f'<span class="xf-tag">City: {esc(c)}</span>' for c in pick_city]
    st.markdown(f'<div class="xf" style="margin:.6rem 0 .2rem">💡 Click bars in the category or city charts to '
                f'cross-filter the page. {"Active: " + " ".join(tags) if tags else ""}</div>', unsafe_allow_html=True)

    if df.empty:
        st.info('No orders match these filters. Widen the period or select ↺ Reset.')
        st.stop()

    # ----- monthly series for sparklines + deltas
    def series(d, freq):
        g = (d.groupby(d['order_date'].dt.to_period(freq))
              .agg(revenue=('net_revenue', 'sum'), orders=('order_id', 'size'),
                   buyers=('customer_id', 'nunique'), ret=('returned', 'mean')).reset_index())
        g['aov'] = g['revenue'] / g['orders']
        g['ts'] = g['order_date'].dt.start_time
        return g

    mm = series(df, 'M')

    def mom(col):
        if len(mm) < 2 or not mm[col].iloc[-2]:
            return None
        return mm[col].iloc[-1] / mm[col].iloc[-2] - 1

    rev = float(df['net_revenue'].sum()); n_orders = len(df); buyers = df['customer_id'].nunique()
    aov = rev / n_orders if n_orders else 0; ret_rate = float(df['returned'].mean())

    st.write('')
    k = st.columns(5)
    kpi(k[0], 'rev', '💰', BLUE, 'Net revenue', f'PKR {compact(rev)}', f'PKR {rev:,.0f}', mm['revenue'], mom('revenue'), delay=0)
    kpi(k[1], 'ord', '📦', NAVY, 'Valid orders', f'{n_orders:,}', 'Order lines in view', mm['orders'], mom('orders'), delay=.05)
    kpi(k[2], 'cus', '👥', ORANGE, 'Customers', f'{buyers:,}', f'of {total_customers:,} registered', mm['buyers'], mom('buyers'), delay=.1)
    kpi(k[3], 'aov', '🧾', VIOLET, 'Avg. revenue / order', f'PKR {compact(aov)}', f'PKR {aov:,.0f}', mm['aov'], mom('aov'), delay=.15)
    kpi(k[4], 'ret', '↩️', PINK, 'Return rate', f'{ret_rate:.1%}', 'Share of lines returned', mm['ret'], mom('ret'),
        lower_is_better=True, delay=.2)
    if len(mm) >= 2:
        st.caption(f'Badges compare {mm["ts"].iloc[-1]:%b %Y} with {mm["ts"].iloc[-2]:%b %Y}.')

    # ----- insights
    cat_v = view_for_cat.groupby('category', as_index=False)['net_revenue'].sum().sort_values('net_revenue', ascending=False)
    cat_v['share'] = cat_v['net_revenue'] / cat_v['net_revenue'].sum()
    city_v = view_for_city.groupby('city', as_index=False)['net_revenue'].sum().sort_values('net_revenue', ascending=False)
    best_m = mm.loc[mm['revenue'].idxmax()]
    wd = df.groupby(df['order_date'].dt.day_name())['net_revenue'].sum()
    city_df = df.groupby('city')['net_revenue'].sum().sort_values(ascending=False)
    with st.container(key='vis_insights'):
        title('Key insights', 'Updates with every filter and click', accent=YELLOW)
        st.markdown(
            f'<div class="narr">'
            f'<div class="narr-item"><div class="narr-k">🏆 Top category</div><div class="narr-v">{esc(cat_v["category"].iloc[0])}</div>'
            f'<div class="narr-s">{cat_v["share"].iloc[0]:.1%} of revenue in view</div></div>'
            f'<div class="narr-item"><div class="narr-k">📈 Best month</div><div class="narr-v">{best_m["ts"]:%B %Y}</div>'
            f'<div class="narr-s">PKR {best_m["revenue"]:,.0f}</div></div>'
            f'<div class="narr-item"><div class="narr-k">📍 Top city</div><div class="narr-v">{esc(city_df.index[0])}</div>'
            f'<div class="narr-s">{city_df.iloc[0] / rev:.1%} of revenue</div></div>'
            f'<div class="narr-item"><div class="narr-k">🗓️ Busiest weekday</div><div class="narr-v">{wd.idxmax()}</div>'
            f'<div class="narr-s">PKR {wd.max():,.0f} net revenue</div></div>'
            f'</div>', unsafe_allow_html=True)

    st.write('')
    # ----- trend explorer
    METRICS = {'Revenue': ('revenue', ',.0f', 'Net revenue (PKR)', BLUE), 'Orders': ('orders', ',', 'Orders', NAVY),
               'Customers': ('buyers', ',', 'Customers', ORANGE), 'Avg. order': ('aov', ',.0f', 'Avg. revenue per order (PKR)', VIOLET)}
    with st.container(key='vis_trend'):
        t1, t2, t3 = st.columns([2, 2.4, 1.4], vertical_alignment='center')
        title('Trend explorer', 'Dashed line is the 3-period moving average', target=t1)
        with t2:
            metric = st.radio('Metric', list(METRICS), horizontal=True, label_visibility='collapsed', key='seg_metric')
        with t3:
            grain = st.radio('Grain', ['Monthly', 'Weekly'], horizontal=True, label_visibility='collapsed', key='seg_grain')
        col, fmt, label, color = METRICS[metric]
        ts = series(df, 'M' if grain == 'Monthly' else 'W')
        ts['ma'] = ts[col].rolling(3, min_periods=1).mean()
        date_fmt = '%b %Y' if grain == 'Monthly' else 'Week of %d %b %Y'
        xax = alt.X('ts:T', title=None, axis=alt.Axis(format='%b %y', labelAngle=0, tickCount=8))
        hover = alt.selection_point(fields=['ts'], nearest=True, on='pointerover', empty=False)
        basec = alt.Chart(ts).encode(x=xax)
        area = basec.mark_area(interpolate='monotone', line={'color': color, 'strokeWidth': 2.6},
                               color=alt.Gradient(gradient='linear', x1=1, x2=1, y1=1, y2=0,
                                                  stops=[alt.GradientStop(color=f'{color}05', offset=0),
                                                         alt.GradientStop(color=f'{color}55', offset=1)])).encode(
            y=alt.Y(f'{col}:Q', title=None, axis=alt.Axis(format='~s')))
        ma = basec.mark_line(interpolate='monotone', strokeDash=[5, 4], strokeWidth=1.8, color=ORANGE if color != ORANGE else NAVY).encode(y='ma:Q')
        rule = basec.mark_rule(color=SUB, strokeDash=[3, 3]).encode(opacity=alt.condition(hover, alt.value(.6), alt.value(0)))
        pts = basec.mark_circle(size=80, color=color, stroke='#fff', strokeWidth=2).encode(
            y=f'{col}:Q', opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=[alt.Tooltip('ts:T', title='Period', format=date_fmt), alt.Tooltip(f'{col}:Q', title=label, format=fmt),
                     alt.Tooltip('ma:Q', title='3-period average', format=fmt)]).add_params(hover)
        show((area + ma + rule + pts).properties(height=300))

    st.write('')
    # ----- clickable category + city charts
    r1, r2 = st.columns(2)
    cat_scale = alt.Scale(domain=list(cat_v['category']), range=(PALETTE * 3)[:len(cat_v)])
    hh = max(280, 34 * min(len(cat_v), 12))
    with r1, st.container(key='vis_cat'):
        title('Revenue by category', 'Click a bar to filter · shift-click for several', accent=PURPLE)
        sel_c = alt.selection_point(name='cat_pick', fields=['category'])
        ch = alt.Chart(cat_v).mark_bar(height=22, cornerRadiusEnd=5, cursor='pointer').encode(
            y=alt.Y('category:N', sort='-x', title=None),
            x=alt.X('net_revenue:Q', title=None, axis=alt.Axis(format='~s')),
            color=alt.Color('category:N', scale=cat_scale, legend=None),
            opacity=alt.condition(sel_c, alt.value(1), alt.value(.3)),
            tooltip=[alt.Tooltip('category:N', title='Category'), alt.Tooltip('net_revenue:Q', title='Net revenue (PKR)', format=',.0f'),
                     alt.Tooltip('share:Q', title='Share', format='.1%')]).add_params(sel_c).properties(height=hh)
        show(ch, on_select='rerun', key='cat_chart')

    with r2, st.container(key='vis_city'):
        topc = city_v.head(10)
        title('Revenue by city', 'Top 10 · click a column to filter', accent=NAVY)
        sel_t = alt.selection_point(name='city_pick', fields=['city'])
        cc = alt.Chart(topc).mark_bar(size=26, cornerRadiusTopLeft=5, cornerRadiusTopRight=5, cursor='pointer').encode(
            x=alt.X('city:N', sort='-y', title=None, axis=alt.Axis(labelAngle=-30)),
            y=alt.Y('net_revenue:Q', title=None, axis=alt.Axis(format='~s')),
            color=alt.Color('net_revenue:Q', legend=None, scale=alt.Scale(range=['#9CC9FF', NAVY])),
            opacity=alt.condition(sel_t, alt.value(1), alt.value(.3)),
            tooltip=[alt.Tooltip('city:N', title='City'), alt.Tooltip('net_revenue:Q', title='Net revenue (PKR)', format=',.0f')]
        ).add_params(sel_t).properties(height=hh)
        show(cc, on_select='rerun', key='city_chart')

    st.write('')
    # ----- heatmap + donut
    r3, r4 = st.columns([2, 1])
    with r3, st.container(key='vis_heat'):
        title('When customers buy', 'Net revenue by weekday and month', accent=ORANGE)
        hm = df.assign(month=df['order_date'].dt.to_period('M').dt.start_time, weekday=df['order_date'].dt.day_name())
        hm = hm.groupby(['month', 'weekday'], as_index=False)['net_revenue'].sum()
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heat = alt.Chart(hm).mark_rect(cornerRadius=4, stroke='#fff', strokeWidth=2).encode(
            x=alt.X('yearmonth(month):O', title=None, axis=alt.Axis(format='%b %y', labelAngle=-35)),
            y=alt.Y('weekday:N', sort=days, title=None, axis=alt.Axis(labelExpr='slice(datum.label, 0, 3)')),
            color=alt.Color('net_revenue:Q', legend=None, scale=alt.Scale(range=['#FFF4E8', '#F6B37E', ORANGE, '#9C3A12'])),
            tooltip=[alt.Tooltip('yearmonth(month):T', title='Month', format='%B %Y'), alt.Tooltip('weekday:N', title='Weekday'),
                     alt.Tooltip('net_revenue:Q', title='Net revenue (PKR)', format=',.0f')]).properties(height=260)
        show(heat)

    with r4, st.container(key='vis_donut'):
        title('Category mix', 'Share of revenue in view', accent=PINK)
        mix = df.groupby('category', as_index=False)['net_revenue'].sum()
        mix['share'] = mix['net_revenue'] / mix['net_revenue'].sum()
        dn = alt.Chart(mix).mark_arc(innerRadius=62, outerRadius=100, cornerRadius=4, padAngle=.015).encode(
            theta='net_revenue:Q', color=alt.Color('category:N', scale=cat_scale, legend=alt.Legend(orient='bottom', columns=2)),
            order=alt.Order('net_revenue:Q', sort='descending'),
            tooltip=[alt.Tooltip('category:N', title='Category'), alt.Tooltip('share:Q', title='Share', format='.1%')])
        ctr = pd.DataFrame({'a': [f'{len(mix)}'], 'b': ['categories']})
        c1t = alt.Chart(ctr).mark_text(fontSize=24, fontWeight=800, color=TEXT, dy=-6).encode(text='a:N')
        c2t = alt.Chart(ctr).mark_text(fontSize=11, color=SUB, dy=15).encode(text='b:N')
        show((dn + c1t + c2t).properties(height=300))

    st.write('')
    # ----- top customers (searchable)
    with st.container(key='vis_table'):
        t1, t2, t3 = st.columns([2.2, 2, 1.2], vertical_alignment='center')
        title('Top customers', 'Ranked by total net spending', accent=GOOD, target=t1)
        q = t2.text_input('Search', placeholder='🔍 Search customer or city', label_visibility='collapsed')
        n_top = t3.selectbox('Show', [10, 25, 50], format_func=lambda n: f'Top {n}', label_visibility='collapsed')
        top = (df.groupby('customer_id')
                 .agg(name=('customer_name', 'first'), city=('city', 'first'),
                      orders=('order_id', 'nunique'), spend=('net_revenue', 'sum'))
                 .sort_values('spend', ascending=False).reset_index(drop=True))
        top['rank'] = np.arange(1, len(top) + 1)
        if q.strip():
            ql = q.strip().lower()
            top = top[top['name'].str.lower().str.contains(ql, regex=False) | top['city'].str.lower().str.contains(ql, regex=False)]
        top = top.head(n_top)
        if top.empty:
            st.markdown('<div class="empty">No customers match that search.</div>', unsafe_allow_html=True)
        else:
            mx = top['spend'].max() or 1
            rows = ''
            for _, r in top.iterrows():
                rk = int(r['rank']); rc = f' r{rk}' if rk <= 3 else ''
                av = PALETTE[sum(map(ord, str(r['name']))) % len(PALETTE)]
                rows += (f'<tr><td><span class="rank{rc}">{rk}</span></td>'
                         f'<td><div class="who"><span class="avatar" style="background:{av}">{esc(initials(r["name"]))}</span>{esc(r["name"])}</div></td>'
                         f'<td>{esc(r["city"])}</td><td class="num">{int(r["orders"]):,}</td>'
                         f'<td><div class="bar"><div class="bar-track"><div class="bar-fill" style="width:{r["spend"] / mx * 100:.1f}%"></div></div>'
                         f'<span class="bar-val">PKR {r["spend"]:,.0f}</span></div></td>'
                         f'<td class="num">{r["spend"] / rev:.2%}</td></tr>')
            st.markdown(f'<div style="overflow-x:auto"><table class="tbl"><thead><tr><th>#</th><th>Customer</th><th>City</th>'
                        f'<th class="num">Orders</th><th>Total spending</th><th class="num">Share</th></tr></thead>'
                        f'<tbody>{rows}</tbody></table></div>', unsafe_allow_html=True)


# ============================================================== CHURN
elif page == 'Churn Prediction':
    DEFAULTS = dict(in_orders=10, in_spend=50000.0, in_aov=5000.0, in_days=30, in_ret=0.05, in_deliv=4.0,
                    in_age=30, in_member='Standard', in_auto=True)
    PRESETS = {
        '🌟 Loyal regular': dict(in_orders=40, in_spend=400000.0, in_days=12, in_ret=0.02, in_deliv=3.0, in_age=35, in_member='Gold'),
        '⚠️ Drifting away': dict(in_orders=6, in_spend=30000.0, in_days=200, in_ret=0.25, in_deliv=8.0, in_age=29, in_member='Standard'),
        '🆕 New customer': dict(in_orders=1, in_spend=4000.0, in_days=20, in_ret=0.0, in_deliv=4.0, in_age=24, in_member='Standard'),
    }
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)

    def apply(values):
        st.session_state.update({**DEFAULTS, **values})

    st.markdown('<div class="page-title">Churn prediction</div><div class="page-sub">Adjust the customer profile and '
                'the prediction updates instantly. Start from a sample profile or enter your own.</div>',
                unsafe_allow_html=True)
    p = st.columns([1, 1, 1, .8, 2.2])
    for col, (name, vals) in zip(p, PRESETS.items()):
        col.button(name, on_click=apply, args=(vals,), use_container_width=True)
    p[3].button('↺ Reset', on_click=apply, args=({},), use_container_width=True)

    left, right = st.columns([1.35, 1], gap='medium')
    with left, st.container(key='vis_inputs'):
        title('Customer profile', accent=PURPLE)
        st.markdown('<div class="hint" style="margin:.3rem 0 -.2rem">🛒 Purchase behaviour</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        total_orders = a.number_input('Total orders', 1, 1000, key='in_orders')
        total_spending = b.number_input('Total spending (PKR)', 0.0, 10000000.0, step=1000.0, key='in_spend')
        auto = st.toggle('Work out average order value from spending ÷ orders', key='in_auto')
        if auto:
            aov = min(total_spending / total_orders, 10000000.0)
            st.markdown(f'<div class="hint">Average order value: <b>PKR {aov:,.0f}</b></div>', unsafe_allow_html=True)
        else:
            aov = st.number_input('Average order value (PKR)', 0.0, 10000000.0, step=500.0, key='in_aov')
        st.markdown('<div class="hint" style="margin:.8rem 0 -.2rem">🚚 Service experience</div>', unsafe_allow_html=True)
        days = st.slider('Days since last order', 0, 2000, key='in_days', help='How long ago the customer last ordered')
        c, d = st.columns(2)
        return_rate = c.slider('Return rate', 0.0, 1.0, step=0.01, key='in_ret', help='0.05 means 5% of items were returned')
        delivery = d.slider('Average delivery days', 0.0, 30.0, step=0.5, key='in_deliv')
        st.markdown('<div class="hint" style="margin:.8rem 0 -.2rem">🪪 Customer profile</div>', unsafe_allow_html=True)
        e, f = st.columns([1, 1.6])
        age = e.number_input('Age', 18, 100, key='in_age')
        with f:
            membership = st.radio('Membership type', ['Standard', 'Silver', 'Gold', 'Premium'], horizontal=True, key='in_member')

    row = {'total_orders': total_orders, 'total_spending': total_spending, 'average_order_value': aov,
           'days_since_last_order': days, 'return_rate': return_rate, 'average_delivery_days': delivery,
           'age': age, 'membership_type': membership}
    X = pd.DataFrame([row])
    model = models['random_forest']; prob = float(model.predict_proba(X)[:, 1][0]); pred = int(prob >= 0.5)
    color = BAD if pred else GOOD

    with right, st.container(key='vis_result'):
        title('Churn probability', 'Random Forest · threshold 50%', accent=color)
        st.markdown(
            f'<div class="gauge"><div class="gauge-arc"></div>'
            f'<div class="gauge-needle" style="transform:rotate({prob * 180 - 90:.1f}deg)"></div><div class="gauge-hub"></div></div>'
            f'<div class="gauge-scale"><span>Low risk</span><span>High risk</span></div>'
            f'<div class="big" style="color:{color}">{prob:.1%}</div>'
            f'<div class="status" style="background:{color}1A;color:{color}">'
            f'{"⚠️ Likely to churn" if pred else "✅ Likely to stay active"}</div>', unsafe_allow_html=True)

        names = {'random_forest': 'Random Forest', 'logistic_regression': 'Logistic Regression', 'neural_network': 'Neural Network'}
        bars = ''
        for key, nm in names.items():
            try:
                pv = float(models[key].predict_proba(X)[:, 1][0])
            except Exception:
                continue
            cl = BAD if pv >= .5 else GOOD
            bars += (f'<div class="mrow"><span>{nm}{" ★" if key == "random_forest" else ""}</span>'
                     f'<div class="mtrack"><div class="mfill" style="width:{pv * 100:.1f}%;background:{cl}"></div>'
                     f'<div class="mmark"></div></div><span class="mval">{pv:.0%}</span></div>')
        if bars:
            st.markdown(f'<div class="vis-sub" style="margin-top:1rem">Model agreement (★ primary model)</div>{bars}',
                        unsafe_allow_html=True)

    st.write('')
    # ----- what-if explorer
    with st.container(key='vis_whatif'):
        w1, w2 = st.columns([2, 1.4], vertical_alignment='center')
        title('What-if explorer', 'How the churn probability changes if one input changes and the rest stay the same',
              accent=BLUE, target=w1)
        FEAT = {'Days since last order': ('days_since_last_order', 0, min(2000, max(365, days * 2)), True, '{:,.0f} days'),
                'Return rate': ('return_rate', 0.0, 1.0, False, '{:.0%}'),
                'Average delivery days': ('average_delivery_days', 0.0, 30.0, False, '{:.1f} days'),
                'Total orders': ('total_orders', 1, min(1000, max(50, total_orders * 2)), True, '{:,.0f} orders'),
                'Age': ('age', 18, 100, True, '{:.0f} years')}
        with w2:
            fname = st.selectbox('Input to vary', list(FEAT), label_visibility='collapsed')
        fcol, lo, hi, is_int, vfmt = FEAT[fname]
        grid = np.linspace(lo, hi, 41)
        if is_int:
            grid = np.unique(np.round(grid).astype(int))
        sweep = pd.DataFrame([row] * len(grid)); sweep[fcol] = grid
        try:
            sp = model.predict_proba(sweep)[:, 1]
            sw = pd.DataFrame({'value': grid, 'prob': sp})
            cur = pd.DataFrame({'value': [row[fcol]], 'prob': [prob]})
            ln = alt.Chart(sw).mark_area(interpolate='monotone', line={'color': BLUE, 'strokeWidth': 2.5},
                                         color=alt.Gradient(gradient='linear', x1=1, x2=1, y1=1, y2=0,
                                                            stops=[alt.GradientStop(color='#118DFF05', offset=0),
                                                                   alt.GradientStop(color='#118DFF40', offset=1)])).encode(
                x=alt.X('value:Q', title=fname), y=alt.Y('prob:Q', title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%')),
                tooltip=[alt.Tooltip('value:Q', title=fname, format=',.2f'), alt.Tooltip('prob:Q', title='Churn probability', format='.1%')])
            thr = alt.Chart(pd.DataFrame({'y': [.5]})).mark_rule(color=BAD, strokeDash=[5, 4]).encode(y='y:Q')
            dot = alt.Chart(cur).mark_point(filled=True, size=160, color=color, stroke='#fff', strokeWidth=2).encode(
                x='value:Q', y='prob:Q', tooltip=[alt.Tooltip('prob:Q', title='Current', format='.1%')])
            show((ln + thr + dot).properties(height=260))

            flip = sw[sw['prob'] < .5] if pred else sw[sw['prob'] >= .5]
            if not flip.empty:
                v = flip.loc[(flip['value'] - row[fcol]).abs().idxmin(), 'value']
                msg = (f'Changing <b>{fname.lower()}</b> to about <b>{vfmt.format(v)}</b> would bring the predicted churn '
                       f'below 50%.' if pred else
                       f'Predicted churn would pass 50% if <b>{fname.lower()}</b> reached about <b>{vfmt.format(v)}</b>.')
                bg, fg = ('#E6F6EC', '#12803B') if pred else ('#FFF6E0', '#8A6400')
            else:
                msg = f'Within this range, <b>{fname.lower()}</b> alone does not change the prediction.'
                bg, fg = SURF2, SUB
            st.markdown(f'<div class="callout" style="background:{bg};color:{fg}">💡 {msg}</div>', unsafe_allow_html=True)
        except Exception as err:
            st.warning(f'The what-if chart could not be calculated: {err}')

    st.caption('This is a model estimate, not a guarantee. The model was trained using purchase history through '
               '31 May 2026 and churn labels from 1 June–31 August 2026. Sample profiles are illustrative.')


# ============================================================== SENTIMENT
else:
    SENT = {'Positive': ('😊', GOOD), 'Neutral': ('😐', MID), 'Negative': ('😞', BAD)}
    EXAMPLES = {
        '😊 Happy': 'The product quality is excellent and delivery was fast. Will order again!',
        '😐 Mixed': 'The item is okay. It matches the description but nothing special.',
        '😞 Unhappy': 'Arrived late and the packaging was damaged. Very disappointed with the quality.',
    }
    mdl = models['sentiment_model']

    st.markdown('<div class="page-title">Review sentiment</div><div class="page-sub">Classify customer reviews as '
                'positive, neutral or negative — one at a time or a whole file at once.</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(['✍️  Single review', '📁  Batch from CSV'])

    with tab1:
        st.session_state.setdefault('review', '')

        def use_example(t):
            st.session_state['review'] = t

        left, right = st.columns([1.4, 1], gap='medium')
        with left, st.container(key='vis_review'):
            title('Customer review', 'Try an example or write your own', accent=BLUE)
            ex = st.columns(len(EXAMPLES) + 1)
            for col, (label, t) in zip(ex, EXAMPLES.items()):
                col.button(label, on_click=use_example, args=(t,), use_container_width=True)
            ex[-1].button('✕ Clear', on_click=use_example, args=('',), use_container_width=True)
            text = st.text_area('Review text', key='review', height=200, label_visibility='collapsed',
                                placeholder='Example: The product quality is excellent and delivery was fast.')
            words = len(text.split())
            st.markdown(f'<div class="hint">{words} word{"s" if words != 1 else ""} · the result updates when you press '
                        f'Ctrl+Enter or click outside the box</div>', unsafe_allow_html=True)

        with right, st.container(key='vis_sent_result'):
            title('Result', accent=MID)
            if text.strip():
                pred = mdl.predict([text.strip()])[0]
                probs = mdl.predict_proba([text.strip()])[0]; classes = mdl.classes_
                conf = float(probs[list(classes).index(pred)])
                emoji, color = SENT.get(str(pred), ('💬', BLUE))
                st.markdown(
                    f'<div class="ring" style="background:conic-gradient({color} {conf * 360:.0f}deg, #EEF0F6 0)">'
                    f'<div class="ring-in"><div><div class="ring-emoji">{emoji}</div><div class="ring-pct">{conf:.0%}</div></div></div></div>'
                    f'<div class="big" style="color:{color};font-size:1.9rem">{esc(pred)}</div>'
                    f'<div class="hint" style="text-align:center;margin-bottom:.8rem">Confidence {conf:.1%}</div>',
                    unsafe_allow_html=True)
                rows = ''
                for c_, p_ in sorted(zip(classes, probs), key=lambda z: -z[1]):
                    cl = SENT.get(str(c_), ('', BLUE))[1]
                    rows += (f'<div class="mrow" style="grid-template-columns:90px 1fr 50px"><span>{esc(c_)}</span>'
                             f'<div class="mtrack"><div class="mfill" style="width:{p_ * 100:.1f}%;background:{cl}"></div></div>'
                             f'<span class="mval">{p_:.0%}</span></div>')
                st.markdown(rows, unsafe_allow_html=True)
            else:
                st.markdown('<div class="empty"><div class="empty-icon">💬</div>Write a review or pick an example '
                            'to see its sentiment.</div>', unsafe_allow_html=True)

    with tab2:
        with st.container(key='vis_batch'):
            title('Analyse many reviews', 'Upload a CSV with one review per row', accent=VIOLET)
            up = st.file_uploader('CSV file', type=['csv'], label_visibility='collapsed')
            if up is None:
                st.markdown('<div class="empty"><div class="empty-icon">📁</div>Upload a CSV file. You will pick the '
                            'column that holds the review text next.</div>', unsafe_allow_html=True)
            else:
                try:
                    data = pd.read_csv(up)
                except Exception as err:
                    st.error(f'This file could not be read as CSV: {err}')
                    st.stop()
                text_cols = [c for c in data.columns if data[c].dtype == object] or list(data.columns)
                guess = next((c for c in text_cols if any(w in c.lower() for w in ('review', 'text', 'comment'))), text_cols[0])
                tc = st.selectbox('Review text column', text_cols, index=text_cols.index(guess))
                texts = data[tc].fillna('').astype(str).str.strip()
                mask = texts != ''
                if not mask.any():
                    st.warning('That column has no text. Pick a different column.')
                else:
                    res = data[mask].copy()
                    res['predicted_sentiment'] = mdl.predict(texts[mask].tolist())
                    res['confidence'] = mdl.predict_proba(texts[mask].tolist()).max(axis=1)
                    dist = res['predicted_sentiment'].astype(str).value_counts().rename_axis('sentiment').reset_index(name='n')
                    dist['share'] = dist['n'] / dist['n'].sum()

                    kc = st.columns(4)
                    kpi(kc[0], 'b0', '🧮', BLUE, 'Reviews analysed', f'{len(res):,}')
                    for i, sname in enumerate(['Positive', 'Neutral', 'Negative']):
                        n = int(dist.loc[dist['sentiment'] == sname, 'n'].sum())
                        e_, cl = SENT[sname]
                        kpi(kc[i + 1], f'b{i + 1}', e_, cl, sname, f'{n:,}', f'{n / len(res):.1%} of reviews')

                    st.write('')
                    g1, g2 = st.columns([1, 1.5])
                    with g1:
                        dom = list(dist['sentiment'])
                        dd = alt.Chart(dist).mark_arc(innerRadius=60, outerRadius=100, cornerRadius=4, padAngle=.02).encode(
                            theta='n:Q', color=alt.Color('sentiment:N', legend=alt.Legend(orient='bottom'),
                                                         scale=alt.Scale(domain=dom, range=[SENT.get(s, ('', BLUE))[1] for s in dom])),
                            tooltip=[alt.Tooltip('sentiment:N', title='Sentiment'), alt.Tooltip('n:Q', title='Reviews', format=','),
                                     alt.Tooltip('share:Q', title='Share', format='.1%')])
                        show(dd.properties(height=260))
                    with g2:
                        prev = res[[tc, 'predicted_sentiment', 'confidence']].head(12)
                        rws = ''
                        for _, r in prev.iterrows():
                            cl = SENT.get(str(r['predicted_sentiment']), ('', BLUE))[1]
                            txt = str(r[tc]); txt = txt[:110] + '…' if len(txt) > 110 else txt
                            rws += (f'<tr><td>{esc(txt)}</td><td><span class="badge" style="background:{cl}1F;color:{cl}">'
                                    f'{esc(r["predicted_sentiment"])}</span></td><td class="num">{r["confidence"]:.0%}</td></tr>')
                        st.markdown(f'<div class="vis-sub">First {len(prev)} results</div><table class="tbl"><thead><tr>'
                                    f'<th>Review</th><th>Sentiment</th><th class="num">Confidence</th></tr></thead>'
                                    f'<tbody>{rws}</tbody></table>', unsafe_allow_html=True)
                    st.download_button('⬇ Download all results (CSV)', res.to_csv(index=False).encode('utf-8'),
                                       'sentiment_results.csv', 'text/csv')

    st.caption('Labels are based on ratings: 1–2 Negative, 3 Neutral, 4–5 Positive.')
