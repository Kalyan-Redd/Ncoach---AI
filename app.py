import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from datetime import datetime, date
import sys

sys.path.insert(0, os.path.dirname(__file__))
from src.database       import (create_user, get_user, get_all_users,
                                 log_meal, get_today_meals, get_today_totals,
                                 get_weekly_daily_totals, get_most_eaten_dishes,
                                 get_all_meals_for_risk, save_risk_scores,
                                 get_latest_risk)
from src.health_rules   import (check_meal_rules, estimate_risk_score,
                                 get_limits_for_user, calculate_tdee)
from src.food_vision    import (predict_food, predict_food_demo,
                                is_model_available, is_yolo_available,
                                DEMO_DISHES)
from src.nutrition_agent import get_coaching_response


# PAGE CONFIG


st.set_page_config(
    page_title="NutriCoach AI",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# CSS


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #ffffff12 !important; }
[data-testid="stSidebar"] * { color: #9ca3af !important; }
[data-testid="stSidebarContent"] { padding: 1.5rem 1rem !important; }
.main .block-container { padding: 2rem 2.5rem !important; max-width: 100% !important; }

.stat-card { background:#f9fafb; border:1px solid #f1f3f5; border-radius:12px; padding:18px 20px; }
.stat-label { font-size:11px; font-weight:600; letter-spacing:.8px; text-transform:uppercase; color:#9ca3af; margin-bottom:6px; }
.stat-value { font-size:26px; font-weight:600; color:#0d1117; font-family:'DM Mono',monospace; }
.stat-unit  { font-size:13px; color:#6b7280; font-family:'DM Sans',sans-serif; font-weight:400; margin-left:3px; }
.stat-sub   { font-size:11px; color:#9ca3af; margin-top:4px; }
.nu-progress-wrap   { background:#f3f4f6; border-radius:4px; height:6px; margin-top:8px; overflow:hidden; }
.nu-progress-safe   { height:6px; border-radius:4px; background:#10b981; }
.nu-progress-warn   { height:6px; border-radius:4px; background:#f59e0b; }
.nu-progress-danger { height:6px; border-radius:4px; background:#ef4444; }
.nu-progress-over   { height:6px; border-radius:4px; background:#dc2626; }

.alert-safe     { background:#d1fae5; border-left:4px solid #10b981; border-radius:0 10px 10px 0; padding:12px 16px; margin-bottom:8px; color:#065f46; font-size:13px; }
.alert-warning  { background:#fef3c7; border-left:4px solid #f59e0b; border-radius:0 10px 10px 0; padding:12px 16px; margin-bottom:8px; color:#92400e; font-size:13px; }
.alert-danger   { background:#fee2e2; border-left:4px solid #ef4444; border-radius:0 10px 10px 0; padding:12px 16px; margin-bottom:8px; color:#991b1b; font-size:13px; }
.alert-exceeded { background:#fecaca; border-left:4px solid #dc2626; border-radius:0 10px 10px 0; padding:12px 16px; margin-bottom:8px; color:#7f1d1d; font-size:13px; font-weight:500; }

.badge-safe  { background:#d1fae5; color:#065f46; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; }
.badge-warn  { background:#fef3c7; color:#92400e; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; }
.badge-danger{ background:#fee2e2; color:#991b1b; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; }
.badge-info  { background:#dbeafe; color:#1e40af; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; }

.section-header { font-size:15px; font-weight:600; color:#0d1117; margin-bottom:14px; display:flex; align-items:center; gap:8px; }
.section-sub    { font-size:12px; color:#9ca3af; font-weight:400; margin-left:4px; }

.chat-user { background:#f3f4f6; border-radius:12px 12px 4px 12px; padding:10px 14px; font-size:13px; color:#374151; line-height:1.6; margin-bottom:10px; max-width:85%; margin-left:auto; }
.chat-bot  { background:#0d1117; border-radius:12px 12px 12px 4px; padding:12px 16px; font-size:13px; color:#e5e7eb; line-height:1.7; margin-bottom:10px; max-width:95%; }

.condition-ckd       { background:#fce7f3; color:#9d174d; padding:4px 12px; border-radius:20px; font-size:11px; font-weight:600; display:inline-block; }
.condition-diabetes  { background:#fef3c7; color:#92400e; padding:4px 12px; border-radius:20px; font-size:11px; font-weight:600; display:inline-block; }
.condition-hypert    { background:#dbeafe; color:#1e40af; padding:4px 12px; border-radius:20px; font-size:11px; font-weight:600; display:inline-block; }
.condition-none      { background:#d1fae5; color:#065f46; padding:4px 12px; border-radius:20px; font-size:11px; font-weight:600; display:inline-block; }

.stTabs [data-baseweb="tab-list"] { gap:4px; background:#f9fafb; padding:4px; border-radius:10px; border:1px solid #f1f3f5; }
.stTabs [data-baseweb="tab"] { border-radius:8px; padding:8px 20px; font-size:13px; font-weight:500; }
.stTabs [aria-selected="true"] { background:#0d1117 !important; color:white !important; }
div[data-testid="metric-container"] { background:#f9fafb; border:1px solid #f1f3f5; border-radius:12px; padding:14px 18px; }
.stButton > button { border-radius:10px; font-weight:500; font-size:13px; }
.stButton > button[kind="primary"] { background:#0d1117; color:white; border:none; }
.stButton > button[kind="primary"]:hover { background:#1a2332; color:white; }
div[data-testid="stDataFrame"] { border:1px solid #f1f3f5; border-radius:10px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)



# HELPERS


def progress_color(pct):
    if pct < 60:  return "nu-progress-safe"
    if pct < 80:  return "nu-progress-warn"
    if pct < 100: return "nu-progress-danger"
    return "nu-progress-over"

def alert_class(level):
    return {"safe":"alert-safe","warning":"alert-warning",
            "danger":"alert-danger","exceeded":"alert-exceeded"}.get(level,"alert-safe")

def risk_color(s):
    return "#10b981" if s < 30 else "#f59e0b" if s < 60 else "#ef4444"

def risk_label(s):
    return "LOW" if s < 30 else "MODERATE" if s < 60 else "HIGH"

def condition_class(c):
    return {"ckd":"condition-ckd","diabetes":"condition-diabetes",
            "hypertension":"condition-hypert","ckd_diabetes":"condition-ckd"}.get(c,"condition-none")

def stat_card_html(label, value, unit, pct, sub=""):
    pc  = progress_color(pct)
    pcd = min(pct, 100)
    return f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}<span class="stat-unit">{unit}</span></div>
        <div class="stat-sub">{sub}</div>
        <div class="nu-progress-wrap"><div class="{pc}" style="width:{pcd}%"></div></div>
    </div>"""

def nutrient_bar_html(label, consumed, limit, unit):
    pct  = round((consumed / limit * 100), 1) if limit > 0 else 0
    pc   = progress_color(pct)
    pcd  = min(pct, 100)
    lv   = "SAFE" if pct<60 else "WATCH" if pct<80 else "DANGER" if pct<100 else "EXCEEDED"
    bc   = "badge-safe" if pct<60 else "badge-warn" if pct<80 else "badge-danger"
    return f"""
    <div style="margin-bottom:14px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="font-size:13px;color:#374151;font-weight:500;">{label}</span>
            <div style="display:flex;align-items:center;gap:8px;">
                <span style="font-size:12px;color:#6b7280;font-family:'DM Mono',monospace;">{consumed:.0f}/{limit:.0f}{unit}</span>
                <span class="{bc}">{lv}</span>
            </div>
        </div>
        <div class="nu-progress-wrap"><div class="{pc}" style="width:{pcd}%"></div></div>
    </div>"""



# SIDEBAR


with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;padding-bottom:20px;border-bottom:1px solid #ffffff15;margin-bottom:20px;">
        <div style="width:40px;height:40px;background:#10b981;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:20px;">🥗</div>
        <div>
            <div style="color:#fff;font-size:15px;font-weight:600;">NutriCoach AI</div>
            <div style="color:#6b7280;font-size:11px;">Personalised nutrition</div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div style="color:#6b7280;font-size:10px;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;">User Profile</div>', unsafe_allow_html=True)

    all_users = get_all_users()
    mode = st.radio("Mode", ["Select user","New user"],
                    horizontal=True, label_visibility="collapsed")

    if mode == "New user":
        with st.form("new_user_form"):
            name = st.text_input("Full name", placeholder="e.g. Ravi Kumar")
            c1,c2 = st.columns(2)
            age    = c1.number_input("Age",     10, 100, 35)
            sex    = c2.selectbox("Sex",        ["male","female"])
            weight = c1.number_input("Weight (kg)", 30.0, 200.0, 70.0)
            height = c2.number_input("Height (cm)", 100.0, 220.0, 170.0)
            activity  = st.selectbox("Activity", ["sedentary","light","moderate","active","very_active"])
            condition = st.selectbox("Condition", ["none","ckd","diabetes","hypertension","ckd_diabetes"])
            if st.form_submit_button("Create Profile", use_container_width=True, type="primary"):
                if name:
                    uid = create_user(name, age, sex, weight, height, activity, condition)
                    st.success(f"Created! ID: {uid}")
                    st.rerun()
    else:
        if not all_users:
            st.info("No users yet. Create one above.")
            st.stop()

        opts    = {u["name"]: u["user_id"] for u in all_users}
        sel     = st.selectbox("User", list(opts.keys()), label_visibility="collapsed")
        user_id = opts[sel]
        user    = get_user(user_id)
        st.session_state["user"]    = user
        st.session_state["user_id"] = user_id

        cond  = user["condition"].upper().replace("_"," + ")
        ccls  = condition_class(user["condition"])
        tdee  = calculate_tdee(user["age"], user["sex"],
                                user["weight_kg"], user["height_cm"],
                                user["activity"])
        init  = user["name"][0].upper()
        st.markdown(f"""
        <div style="background:#ffffff0a;border:1px solid #ffffff15;border-radius:12px;padding:16px;margin-top:12px;">
            <div style="width:42px;height:42px;border-radius:50%;background:#10b981;display:flex;align-items:center;justify-content:center;font-weight:600;color:#fff;font-size:15px;margin-bottom:10px;">{init}</div>
            <div style="color:#fff;font-size:14px;font-weight:600;">{user['name']}</div>
            <div style="color:#6b7280;font-size:12px;margin-top:2px;">{user['age']}y · {user['sex'].title()} · {user['weight_kg']}kg</div>
            <div style="margin-top:8px;"><span class="{ccls}">⚕ {cond}</span></div>
            <div style="color:#6b7280;font-size:11px;margin-top:8px;">Daily target: {tdee} kcal</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div style="color:#6b7280;font-size:10px;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin:20px 0 8px;">System</div>', unsafe_allow_html=True)
    if is_yolo_available():
        st.markdown('<div style="background:#10b98120;color:#10b981;padding:8px 12px;border-radius:8px;font-size:11px;font-weight:500;">✓ YOLOv8 Multi-Detection</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background:#f59e0b15;color:#f59e0b;padding:8px 12px;border-radius:8px;font-size:11px;">⚠ Demo Mode</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:#4b5563;font-size:11px;margin-top:8px;">{date.today().strftime("%d %B %Y")}</div>', unsafe_allow_html=True)



# GUARD


if "user" not in st.session_state:
    st.markdown("""
    <div style="text-align:center;padding:80px;">
        <div style="font-size:48px;margin-bottom:16px;">🥗</div>
        <div style="font-size:22px;font-weight:600;color:#0d1117;margin-bottom:8px;">Welcome to NutriCoach AI</div>
        <div style="font-size:14px;color:#6b7280;">Select or create a profile in the sidebar to begin.</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

user    = st.session_state["user"]
user_id = st.session_state["user_id"]
limits  = get_limits_for_user(user)


# HEADER


today_totals = get_today_totals(user_id)
rule_result  = check_meal_rules(user, today_totals)
alerts       = rule_result["alerts"]
meal_count   = len(get_today_meals(user_id))
hour         = datetime.now().hour
greeting     = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
exc_count    = sum(1 for a in alerts if a["level"] == "exceeded")
status_txt   = f"⛔ {exc_count} limit(s) exceeded" if exc_count > 0 else \
               f"⚠ {len(alerts)} alert(s) active" if alerts else "✓ All nutrients safe today"
status_col   = "#ef4444" if exc_count > 0 else "#f59e0b" if alerts else "#10b981"

st.markdown(f"""
<div style="background:#0d1117;border-radius:16px;padding:28px 32px;margin-bottom:24px;display:flex;justify-content:space-between;align-items:center;">
    <div>
        <div style="font-size:22px;font-weight:600;color:#fff;">{greeting}, {user['name'].split()[0]} 👋</div>
        <div style="font-size:13px;color:#6b7280;margin-top:4px;">{date.today().strftime('%A, %d %B %Y')} · {meal_count} meal(s) logged today</div>
    </div>
    <div style="background:{status_col}20;border:1px solid {status_col}40;color:{status_col};padding:8px 20px;border-radius:24px;font-size:13px;font-weight:600;">{status_txt}</div>
</div>""", unsafe_allow_html=True)



# TABS


tab1, tab2, tab3, tab4 = st.tabs([
    "📷 Log Meal",
    "📊 Dashboard",
    "🔮 Health Risk",
    "🤖 AI Coach",
])



# TAB 1 — LOG MEAL


with tab1:
    col_l, col_r = st.columns([1,1], gap="large")

    with col_l:
        st.markdown('<div class="section-header">📷 Upload Meal Photo<span class="section-sub">— AI detects each item individually</span></div>', unsafe_allow_html=True)

        mode_inp = st.radio("Input", ["Upload photo","Demo mode"],
                            horizontal=True, label_visibility="collapsed")
        uploaded_file = None
        demo_dish     = None

        if mode_inp == "Upload photo":
            uploaded_file = st.file_uploader("Photo", type=["jpg","jpeg","png"],
                                              label_visibility="collapsed")
            if uploaded_file:
                st.image(uploaded_file, use_column_width=True, caption="Uploaded meal")
        else:
            demo_dish = st.selectbox("Dish", DEMO_DISHES, label_visibility="collapsed")
            st.markdown(f'<div style="background:#f9fafb;border:1px solid #f1f3f5;border-radius:12px;padding:20px;text-align:center;color:#6b7280;font-size:13px;">Demo: <b>{demo_dish}</b></div>', unsafe_allow_html=True)

        meal_type = st.selectbox("Meal type", ["breakfast","lunch","dinner","snack"])
        detect_btn = st.button("🔍 Detect Foods & Calculate Nutrition",
                                type="primary", use_container_width=True)

    with col_r:
        if detect_btn and (uploaded_file or demo_dish):
            with st.spinner("🧠 Analysing your meal..."):
                if uploaded_file and is_model_available():
                    result = predict_food(Image.open(uploaded_file))
                elif demo_dish:
                    result = predict_food_demo(demo_dish, 250)
                else:
                    result = predict_food_demo()

            dets   = result["detections"]
            total  = result["total_nutrition"]
            mused  = result["model_used"]
            icount = result["item_count"]

            if mused == "yolov8" and result.get("annotated_image"):
                st.image(result["annotated_image"],
                         caption=f"YOLOv8 — {icount} item(s) detected",
                         use_column_width=True)

            if mused == "yolov8":
                st.markdown(f'<div class="badge-info">🎯 YOLOv8 · {icount} food item(s) detected</div>', unsafe_allow_html=True)
            elif mused == "efficientnet":
                st.markdown(f'<div class="badge-info">📸 {dets[0]["dish_name"]} · {dets[0]["confidence"]*100:.0f}% confidence</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-warn">⚠ Demo mode</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Individual items table
            if dets:
                st.markdown('<div class="section-header">🍽️ Individual items detected</div>', unsafe_allow_html=True)
                rows = [{"Dish": d["dish_name"],
                         "Conf": f"{d['confidence']*100:.0f}%",
                         "Portion": f"{d['portion_g']}g",
                         "Calories": f"{d['nutrition']['calories']} kcal",
                         "Protein": f"{d['nutrition']['protein_g']}g",
                         "K (mg)": f"{d['nutrition']['potassium_mg']}",
                         "Na (mg)": f"{d['nutrition']['sodium_mg']}"}
                        for d in dets]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            # Total nutrition
            st.markdown('<div class="section-header" style="margin-top:16px;">📊 Total plate nutrition</div>', unsafe_allow_html=True)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("🔥 Calories",  f"{total.get('calories',0):.0f} kcal")
            c2.metric("💪 Protein",   f"{total.get('protein_g',0):.1f} g")
            c3.metric("⚡ Potassium", f"{total.get('potassium_mg',0):.0f} mg")
            c4.metric("🧂 Sodium",    f"{total.get('sodium_mg',0):.0f} mg")

            # Log to database
            for d in dets:
                log_meal(user_id, d["dish_name"], d["confidence"],
                         d["nutrition"], meal_type)

            # Clinical alerts
            fresh_totals = get_today_totals(user_id)
            fresh_rules  = check_meal_rules(user, fresh_totals)
            fresh_alerts = fresh_rules["alerts"]

            st.markdown('<div class="section-header" style="margin-top:16px;">🚨 Clinical alerts</div>', unsafe_allow_html=True)
            if not fresh_alerts:
                st.markdown(f'<div class="alert-safe">✓ {fresh_rules["summary"]}</div>', unsafe_allow_html=True)
            else:
                for a in fresh_alerts:
                    st.markdown(f'<div class="{alert_class(a["level"])}">{a["message"]}</div>', unsafe_allow_html=True)
                    if a.get("clinical_note"):
                        st.caption(f"ℹ️ {a['clinical_note']}")

            st.success(f"✅ {len(dets)} item(s) logged at {datetime.now().strftime('%H:%M')}")

    # Today log
    st.divider()
    st.markdown(f'<div class="section-header">🗓️ Today\'s meal log <span class="section-sub">— {date.today().strftime("%d %B %Y")}</span></div>', unsafe_allow_html=True)
    today_meals = get_today_meals(user_id)
    if today_meals:
        df = pd.DataFrame(today_meals)[["log_time","meal_type","dish_name","calories","protein_g","potassium_mg","phosphorus_mg","sodium_mg"]].copy()
        df.columns = ["Time","Type","Dish","Cal(kcal)","Protein(g)","K(mg)","P(mg)","Na(mg)"]
        st.dataframe(df, hide_index=True, use_container_width=True)
        t = get_today_totals(user_id)
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Calories",   f"{t['calories']:.0f} kcal")
        c2.metric("Protein",    f"{t['protein_g']:.1f} g")
        c3.metric("Potassium",  f"{t['potassium_mg']:.0f} mg")
        c4.metric("Phosphorus", f"{t['phosphorus_mg']:.0f} mg")
        c5.metric("Sodium",     f"{t['sodium_mg']:.0f} mg")
    else:
        st.markdown('<div style="text-align:center;padding:40px;color:#9ca3af;font-size:14px;">No meals logged today. Upload your first meal photo above! 📷</div>', unsafe_allow_html=True)



# TAB 2 — DASHBOARD


with tab2:
    st.markdown('<div class="section-header">📊 Weekly Nutrition Dashboard</div>', unsafe_allow_html=True)

    pcts = rule_result["percentages"]
    cols = st.columns(4)
    for i, (key, label, unit) in enumerate([
        ("calories","Calories","kcal"),
        ("protein_g","Protein","g"),
        ("potassium_mg","Potassium","mg"),
        ("sodium_mg","Sodium","mg"),
    ]):
        d = pcts.get(key, {})
        cols[i].markdown(stat_card_html(
            label, f"{d.get('consumed',0):.0f}", unit,
            d.get("percent",0),
            f"{d.get('percent',0):.0f}% of {d.get('limit',1):.0f}{unit} limit"
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1,1], gap="large")

    with col_a:
        st.markdown('<div class="section-header">Today\'s nutrient progress</div>', unsafe_allow_html=True)
        bars = ""
        for key,label,unit in [
            ("calories","Calories","kcal"),("protein_g","Protein","g"),
            ("carbs_g","Carbohydrates","g"),("fat_g","Fat","g"),
            ("potassium_mg","Potassium","mg"),("phosphorus_mg","Phosphorus","mg"),
            ("sodium_mg","Sodium","mg"),
        ]:
            d = pcts.get(key,{})
            bars += nutrient_bar_html(label, d.get("consumed",0), d.get("limit",1), unit)
        st.markdown(bars, unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-header">Weekly calorie intake</div>', unsafe_allow_html=True)
        weekly_data = get_weekly_daily_totals(user_id, days=7)
        if weekly_data:
            df_w = pd.DataFrame(weekly_data)
            fig  = px.bar(df_w, x="log_date", y="calories",
                          color_discrete_sequence=["#10b981"],
                          labels={"log_date":"Date","calories":"Calories (kcal)"})
            fig.add_hline(y=limits["calories"], line_dash="dash",
                          line_color="#ef4444", line_width=1.5,
                          annotation_text="Daily target")
            fig.update_layout(height=280, margin=dict(t=10,b=10,l=0,r=0),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               font=dict(family="DM Sans",size=12,color="#6b7280"),
                               xaxis=dict(showgrid=False),
                               yaxis=dict(showgrid=True,gridcolor="#f3f4f6"),
                               showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Log meals to see weekly trends.")

    st.markdown('<div class="section-header" style="margin-top:8px;">Most frequently eaten dishes (last 30 days)</div>', unsafe_allow_html=True)
    top = get_most_eaten_dishes(user_id, days=30, limit=8)
    if top:
        df_d = pd.DataFrame(top)
        fig2 = px.bar(df_d, x="dish_name", y="frequency",
                      color="avg_calories",
                      color_continuous_scale=[[0,"#d1fae5"],[0.5,"#10b981"],[1,"#065f46"]],
                      labels={"dish_name":"Dish","frequency":"Times eaten","avg_calories":"Avg cal"})
        fig2.update_layout(height=260, margin=dict(t=10,b=10,l=0,r=0),
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="DM Sans",size=12,color="#6b7280"),
                            xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=True,gridcolor="#f3f4f6"),
                            coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Log more meals to see patterns.")



# TAB 3 — HEALTH RISK


with tab3:
    st.markdown('<div class="section-header">🔮 Future Disease Risk Prediction<span class="section-sub">— Based on 30-day eating patterns</span></div>', unsafe_allow_html=True)
    st.caption("Scores range from 0 (no risk) to 100 (very high risk). Based on Harvard research linking diet to chronic disease.")

    avg_data = get_all_meals_for_risk(user_id, days=30)

    if not avg_data or avg_data.get("days_tracked",0) == 0:
        st.markdown('<div style="text-align:center;padding:60px;background:#f9fafb;border-radius:14px;border:1px solid #f1f3f5;"><div style="font-size:36px;margin-bottom:12px;">📊</div><div style="font-size:15px;font-weight:500;color:#374151;">Not enough data yet</div><div style="font-size:13px;color:#9ca3af;margin-top:6px;">Log at least 3 days of meals to see risk scores.</div></div>', unsafe_allow_html=True)
    else:
        risks = estimate_risk_score(avg_data, user)
        save_risk_scores(user_id, risks["ckd_risk"], risks["diabetes_risk"],
                         risks["heart_risk"], risks["hypert_risk"])
        days = int(avg_data.get("days_tracked",0))
        st.markdown(f'<div style="color:#6b7280;font-size:13px;margin-bottom:20px;">Based on <b style="color:#0d1117">{days} day(s)</b> of meal data</div>', unsafe_allow_html=True)

        # Risk cards
        cols = st.columns(4)
        for i,(label,score) in enumerate([
            ("Kidney Disease",  risks["ckd_risk"]),
            ("Diabetes",        risks["diabetes_risk"]),
            ("Heart Disease",   risks["heart_risk"]),
            ("Hypertension",    risks["hypert_risk"]),
        ]):
            c = risk_color(score)
            l = risk_label(score)
            cols[i].markdown(f"""
            <div style="background:#f9fafb;border:1px solid #f1f3f5;border-radius:14px;padding:20px;text-align:center;">
                <div style="font-size:32px;font-weight:600;color:{c};font-family:'DM Mono',monospace;">{score}</div>
                <div style="font-size:11px;font-weight:600;color:{c};margin:4px 0;">{l}</div>
                <div style="font-size:12px;color:#6b7280;">{label}</div>
                <div style="background:#f3f4f6;border-radius:4px;height:4px;margin-top:12px;overflow:hidden;">
                    <div style="width:{score}%;height:4px;background:{c};border-radius:4px;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_c, col_d = st.columns([1,1], gap="large")

        with col_c:
            st.markdown('<div class="section-header">Risk score overview</div>', unsafe_allow_html=True)
            vals   = [risks["ckd_risk"],risks["diabetes_risk"],risks["heart_risk"],risks["hypert_risk"]]
            labels = ["Kidney","Diabetes","Heart","BP"]
            fig3   = go.Figure(go.Bar(
                x=labels, y=vals,
                marker_color=[risk_color(v) for v in vals],
                text=[f"{v}/100" for v in vals],
                textposition="outside",
                textfont=dict(size=12,family="DM Mono"),
            ))
            fig3.add_hline(y=60, line_dash="dash", line_color="#ef4444",
                           line_width=1.5, annotation_text="High risk threshold")
            fig3.update_layout(height=300, margin=dict(t=30,b=10,l=0,r=0),
                                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(family="DM Sans",size=12,color="#6b7280"),
                                yaxis=dict(range=[0,115],showgrid=True,gridcolor="#f3f4f6"),
                                xaxis=dict(showgrid=False), showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

        with col_d:
            st.markdown('<div class="section-header">🎯 What-if simulator</div>', unsafe_allow_html=True)
            st.caption("Adjust sliders to see how dietary changes reduce your risk.")
            ns = st.slider("Daily sodium (mg)", 500, 3000, int(avg_data.get("avg_daily_sodium",2000)), 50)
            np_ = st.slider("Daily protein (g)", 20, 120, int(avg_data.get("avg_daily_protein",60)), 5)
            nc = st.slider("Daily calories (kcal)", 1200, 3000, int(avg_data.get("avg_daily_calories",2000)), 50)
            sim = avg_data.copy()
            sim.update({"avg_daily_sodium":ns,"avg_daily_protein":np_,"avg_daily_calories":nc})
            sr  = estimate_risk_score(sim, user)
            s1,s2,s3,s4 = st.columns(4)
            for col,lbl,orig,simr in [
                (s1,"Kidney",risks["ckd_risk"],sr["ckd_risk"]),
                (s2,"Diab.",risks["diabetes_risk"],sr["diabetes_risk"]),
                (s3,"Heart",risks["heart_risk"],sr["heart_risk"]),
                (s4,"BP",risks["hypert_risk"],sr["hypert_risk"]),
            ]:
                col.metric(lbl, f"{simr}/100", delta=f"{simr-orig:+.0f}", delta_color="inverse")



# TAB 4 — AI COACH


with tab4:
    st.markdown('<div class="section-header">🤖 AI Nutrition Coach<span class="section-sub">— Llama 3.3 + RAG clinical knowledge</span></div>', unsafe_allow_html=True)
    st.caption("Ask anything about your diet, condition, or food choices. The coach uses your full meal history and verified clinical guidelines.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Quick questions
    st.markdown('<div style="font-size:13px;font-weight:500;color:#374151;margin-bottom:10px;">Quick questions:</div>', unsafe_allow_html=True)
    quick_qs = [
        "What should I eat for dinner?",
        "Which foods should I avoid?",
        "Am I eating too much protein?",
        "Give me a safe meal plan for tomorrow",
        "What is my biggest health risk?",
        "How can I reduce my kidney risk?",
    ]
    qcols = st.columns(3)
    sel_q = None
    for i,q in enumerate(quick_qs):
        if qcols[i%3].button(q, key=f"qq_{i}", use_container_width=True):
            sel_q = q

    st.markdown("<br>", unsafe_allow_html=True)

    # Chat history display
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    # Input
    user_inp = st.chat_input("Ask your nutrition coach...")
    final_q  = sel_q or user_inp

    if final_q:
        st.session_state.chat_history.append({"role":"user","content":final_q})
        st.markdown(f'<div class="chat-user">👤 {final_q}</div>', unsafe_allow_html=True)

        with st.spinner("🤔 Analysing your meals and generating personalised advice..."):
            fresh_totals = get_today_totals(user_id)
            fresh_rules  = check_meal_rules(user, fresh_totals)
            weekly_avg   = get_all_meals_for_risk(user_id, days=7)
            meals_today  = get_today_meals(user_id)
            last_meal    = meals_today[-1] if meals_today else None

            response = get_coaching_response(
                user=user,
                today_totals=fresh_totals,
                alerts=fresh_rules["alerts"],
                rule_result=fresh_rules,
                latest_meal=last_meal,
                weekly_summary=weekly_avg,
                user_question=final_q,
            )

        st.markdown(f'<div class="chat-bot">🤖 {response}</div>', unsafe_allow_html=True)
        st.session_state.chat_history.append({"role":"assistant","content":response})

    if st.session_state.chat_history:
        if st.button("🗑️ Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()