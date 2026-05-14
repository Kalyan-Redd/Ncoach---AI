# DAILY NUTRIENT LIMITS PER CONDITION
# Source: KDIGO 2024, ADA 2024, WHO Hypertension Guidelines


LIMITS = {
    "none": {
        "calories":      2000,
        "protein_g":     50,
        "carbs_g":       225,
        "fat_g":         65,
        "potassium_mg":  3500,
        "phosphorus_mg": 1200,
        "sodium_mg":     2300,
        "fiber_g":       25,
    },
    "ckd": {
        "calories":      2000,
        "protein_g":     40,       # 0.6g/kg for 70kg person
        "carbs_g":       250,
        "fat_g":         65,
        "potassium_mg":  2000,     # strict limit
        "phosphorus_mg": 800,      # strict limit
        "sodium_mg":     1500,     # strict limit
        "fiber_g":       25,
    },
    "diabetes": {
        "calories":      1800,
        "protein_g":     50,
        "carbs_g":       130,      # strict carb limit
        "fat_g":         55,
        "potassium_mg":  3500,
        "phosphorus_mg": 1200,
        "sodium_mg":     2300,
        "fiber_g":       30,       # high fiber helps glucose
    },
    "hypertension": {
        "calories":      2000,
        "protein_g":     50,
        "carbs_g":       225,
        "fat_g":         65,
        "potassium_mg":  3500,
        "phosphorus_mg": 1200,
        "sodium_mg":     1500,     # strict sodium limit
        "fiber_g":       25,
    },
    "ckd_diabetes": {
        "calories":      1800,
        "protein_g":     40,       # strictest protein
        "carbs_g":       130,      # strict carbs
        "fat_g":         55,
        "potassium_mg":  2000,     # strict potassium
        "phosphorus_mg": 800,      # strict phosphorus
        "sodium_mg":     1500,     # strict sodium
        "fiber_g":       25,
    },
}

# Personalise protein limit based on user weight
def get_protein_limit(condition, weight_kg):
    """
    CKD: 0.6-0.8g per kg body weight
    Others: 0.8g per kg body weight
    """
    if condition in ("ckd", "ckd_diabetes"):
        return round(0.7 * weight_kg, 1)   # 0.7g/kg for CKD
    return round(0.8 * weight_kg, 1)



# HARRIS-BENEDICT CALORIE CALCULATOR

ACTIVITY_FACTORS = {
    "sedentary":  1.2,
    "light":      1.375,
    "moderate":   1.55,
    "active":     1.725,
    "very_active":1.9,
}

def calculate_tdee(age, sex, weight_kg, height_cm, activity="moderate"):
    """
    Calculate Total Daily Energy Expenditure using Harris-Benedict.
    Returns calorie target as int.
    """
    if sex == "male":
        bmr = 88.36 + (13.4 * weight_kg) + (4.8 * height_cm) - (5.7 * age)
    else:
        bmr = 447.6 + (9.25 * weight_kg) + (3.1 * height_cm) - (4.33 * age)

    factor = ACTIVITY_FACTORS.get(activity, 1.55)
    return round(bmr * factor)


def get_limits_for_user(user):
    """
    Get personalised daily limits for a user dict.
    Adjusts protein and calories based on weight/age/sex.
    """
    condition = user.get("condition", "none")
    limits = LIMITS.get(condition, LIMITS["none"]).copy()

    # Personalise protein
    weight = user.get("weight_kg", 70)
    limits["protein_g"] = get_protein_limit(condition, weight)

    # Personalise calories
    tdee = calculate_tdee(
        age=user.get("age", 30),
        sex=user.get("sex", "male"),
        weight_kg=weight,
        height_cm=user.get("height_cm", 170),
        activity=user.get("activity", "moderate")
    )
    limits["calories"] = tdee

    return limits

# ALERT THRESHOLDS
# 

ALERT_LEVELS = {
    "safe":     (0,   60),    # green  — below 60% of limit
    "warning":  (60,  80),    # yellow — 60-80% of limit
    "danger":   (80,  100),   # orange — 80-100% of limit
    "exceeded": (100, 9999),  # red    — over limit
}

def get_alert_level(pct):
    """Return alert level string based on percentage used."""
    if pct < 60:
        return "safe"
    elif pct < 80:
        return "warning"
    elif pct < 100:
        return "danger"
    else:
        return "exceeded"


# 
# MAIN RULE CHECK FUNCTION
# 

def check_meal_rules(user, today_totals, new_meal_nutrition=None):
    """
    Main function — checks today's nutrient totals against user's limits.

    Parameters:
        user             : dict from database.get_user()
        today_totals     : dict from database.get_today_totals()
        new_meal_nutrition: dict (optional) — nutrition of meal just added

    Returns:
        dict with:
            limits      : personalised daily limits
            percentages : % of each limit used
            alerts      : list of alert dicts
            summary     : overall status string
            safe_swaps  : list of safer food suggestions
    """
    limits = get_limits_for_user(user)
    condition = user.get("condition", "none")

    # Nutrients to check
    nutrients = [
        "calories", "protein_g", "carbs_g", "fat_g",
        "potassium_mg", "phosphorus_mg", "sodium_mg"
    ]

    nutrient_labels = {
        "calories":      "Calories",
        "protein_g":     "Protein",
        "carbs_g":       "Carbohydrates",
        "fat_g":         "Fat",
        "potassium_mg":  "Potassium",
        "phosphorus_mg": "Phosphorus",
        "sodium_mg":     "Sodium",
    }

    nutrient_units = {
        "calories":      "kcal",
        "protein_g":     "g",
        "carbs_g":       "g",
        "fat_g":         "g",
        "potassium_mg":  "mg",
        "phosphorus_mg": "mg",
        "sodium_mg":     "mg",
    }

    percentages = {}
    alerts = []

    for nutrient in nutrients:
        consumed = today_totals.get(nutrient, 0) or 0
        limit    = limits.get(nutrient, 1)
        pct      = round((consumed / limit) * 100, 1) if limit > 0 else 0
        level    = get_alert_level(pct)
        percentages[nutrient] = {
            "consumed": round(consumed, 1),
            "limit":    round(limit, 1),
            "percent":  pct,
            "level":    level,
            "unit":     nutrient_units[nutrient],
            "label":    nutrient_labels[nutrient],
        }

        # Generate alert if warning or above
        if level in ("warning", "danger", "exceeded"):
            alert = _build_alert(
                nutrient, nutrient_labels[nutrient],
                consumed, limit, pct, level,
                nutrient_units[nutrient], condition
            )
            alerts.append(alert)

    # Overall summary
    exceeded = [a for a in alerts if a["level"] == "exceeded"]
    danger   = [a for a in alerts if a["level"] == "danger"]

    if exceeded:
        summary = "🔴 LIMIT EXCEEDED — You have crossed daily limits for: " + \
                  ", ".join(a["nutrient_label"] for a in exceeded)
    elif danger:
        summary = "🟠 APPROACHING LIMIT — Be careful with: " + \
                  ", ".join(a["nutrient_label"] for a in danger)
    elif alerts:
        summary = "🟡 WATCH OUT — Monitor your intake of: " + \
                  ", ".join(a["nutrient_label"] for a in alerts)
    else:
        summary = "🟢 GREAT JOB — All nutrients within healthy limits today!"

    # Safe food suggestions based on condition
    safe_swaps = get_safe_swaps(condition, alerts)

    return {
        "limits":      limits,
        "percentages": percentages,
        "alerts":      alerts,
        "summary":     summary,
        "safe_swaps":  safe_swaps,
        "condition":   condition,
    }


def _build_alert(nutrient, label, consumed, limit, pct, level, unit, condition):
    """Build a single alert dict with message."""
    messages = {
        "exceeded": f"⛔ {label} limit exceeded! You've consumed {consumed:.1f}{unit} "
                    f"out of your {limit:.1f}{unit} daily limit ({pct:.0f}%).",
        "danger":   f"⚠️  {label} is at {pct:.0f}% of your daily limit. "
                    f"({consumed:.1f}/{limit:.1f}{unit}) — be careful with next meal.",
        "warning":  f"💛 {label} at {pct:.0f}% of daily limit "
                    f"({consumed:.1f}/{limit:.1f}{unit}). Stay mindful.",
    }

    # Special messages for CKD critical nutrients
    ckd_special = {
        "potassium_mg": "High potassium can cause dangerous heart rhythm issues in CKD patients.",
        "phosphorus_mg":"High phosphorus weakens bones and stresses the kidneys.",
        "protein_g":    "Excess protein increases kidney workload in CKD patients.",
        "sodium_mg":    "High sodium causes fluid retention and raises blood pressure.",
    }

    alert = {
        "nutrient":       nutrient,
        "nutrient_label": label,
        "consumed":       consumed,
        "limit":          limit,
        "percent":        pct,
        "level":          level,
        "unit":           unit,
        "message":        messages.get(level, ""),
        "clinical_note":  ckd_special.get(nutrient, "") if condition in ("ckd","ckd_diabetes") else "",
    }
    return alert



# SAFE FOOD SUGGESTIONS


SAFE_FOODS = {
    "ckd": {
        "potassium_mg":  ["Lauki sabzi (bottle gourd)", "White rice", "Cabbage sabzi",
                          "Arbi (colocasia)", "Bread (white)"],
        "phosphorus_mg": ["White rice", "Egg white curry", "Apple",
                          "Cucumber raita", "Lauki sabzi"],
        "protein_g":     ["Plain rice", "Lauki sabzi", "Cucumber salad",
                          "White bread", "Apple"],
        "sodium_mg":     ["Home-cooked dal (no salt)", "Plain rice",
                          "Steamed idli (no chutney)", "Lauki sabzi"],
    },
    "diabetes": {
        "carbs_g":       ["Palak sabzi", "Methi dal", "Egg bhurji",
                          "Tandoori chicken", "Mixed salad"],
        "calories":      ["Rasam", "Sambar (small portion)", "Steamed idli",
                          "Grilled fish", "Cucumber raita"],
        "sodium_mg":     ["Home-cooked dal", "Steamed vegetables", "Plain curd"],
    },
    "hypertension": {
        "sodium_mg":     ["Steamed idli", "Plain rice with dal",
                          "Lauki sabzi", "Banana", "Plain curd"],
    },
    "none": {
        "calories":      ["Rasam", "Salad", "Steamed vegetables",
                          "Buttermilk", "Fruit"],
        "fat_g":         ["Steamed idli", "Plain dal", "Grilled chicken",
                          "Curd rice", "Fruit salad"],
    }
}

def get_safe_swaps(condition, alerts):
    """Return safe food suggestions based on which nutrients are alerted."""
    if not alerts:
        return ["You are doing great! Keep eating balanced Indian meals."]

    suggestions = []
    condition_foods = SAFE_FOODS.get(condition, SAFE_FOODS["none"])

    for alert in alerts:
        if alert["level"] in ("danger", "exceeded"):
            nutrient = alert["nutrient"]
            foods = condition_foods.get(nutrient, [])
            if foods:
                label = alert["nutrient_label"]
                suggestions.append(
                    f"For high {label}: Try {', '.join(foods[:3])}"
                )

    if not suggestions:
        suggestions = ["Prefer home-cooked meals with less oil and salt for your next meal."]

    return suggestions



# FUTURE RISK SCORE (Rule-Based Estimate)
# Used before ML model is trained


def estimate_risk_score(avg_daily, user):
    """
    Estimate future disease risk score (0-100) from average daily nutrition.
    Higher score = higher risk.

    Parameters:
        avg_daily : dict of average daily nutrient values
        user      : user dict with age, weight, condition

    Returns:
        dict with ckd_risk, diabetes_risk, heart_risk, hypert_risk
    """
    age        = user.get("age", 30)
    condition  = user.get("condition", "none")
    weight     = user.get("weight_kg", 70)
    height     = user.get("height_cm", 170)
    bmi        = weight / ((height/100) ** 2)

    sodium     = avg_daily.get("avg_daily_sodium", 0) or 0
    protein    = avg_daily.get("avg_daily_protein", 0) or 0
    calories   = avg_daily.get("avg_daily_calories", 0) or 0
    carbs      = avg_daily.get("avg_daily_carbs", 0) or 0
    potassium  = avg_daily.get("avg_daily_potassium", 0) or 0

    # ── CKD Risk 
    ckd_risk = 0
    if protein > 60:   ckd_risk += 25   # high protein stresses kidneys
    if sodium  > 2000: ckd_risk += 20
    if potassium > 3000: ckd_risk += 10
    if condition == "diabetes": ckd_risk += 15   # diabetes → CKD risk
    if age > 45:       ckd_risk += 10
    if bmi > 27:       ckd_risk += 10
    ckd_risk = min(ckd_risk, 100)

    # ── Diabetes Risk 
    diabetes_risk = 0
    if carbs > 280:    diabetes_risk += 30   # high carb intake
    if calories > 2500: diabetes_risk += 20
    if bmi > 25:       diabetes_risk += 20
    if age > 40:       diabetes_risk += 10
    if sodium > 2000:  diabetes_risk += 10
    diabetes_risk = min(diabetes_risk, 100)

    # ── Heart Disease Risk 
    heart_risk = 0
    if sodium > 2000:  heart_risk += 25
    if calories > 2500: heart_risk += 20
    if bmi > 27:       heart_risk += 20
    if age > 45:       heart_risk += 15
    if condition in ("diabetes","hypertension"): heart_risk += 15
    heart_risk = min(heart_risk, 100)

    # ── Hypertension Risk 
    hypert_risk = 0
    if sodium > 1800:  hypert_risk += 35
    if bmi > 25:       hypert_risk += 20
    if age > 40:       hypert_risk += 15
    if calories > 2300: hypert_risk += 15
    if condition == "ckd": hypert_risk += 10
    hypert_risk = min(hypert_risk, 100)

    return {
        "ckd_risk":      round(ckd_risk),
        "diabetes_risk": round(diabetes_risk),
        "heart_risk":    round(heart_risk),
        "hypert_risk":   round(hypert_risk),
    }



# QUICK TEST — run: python src/health_rules.py

if __name__ == "__main__":
    # Simulate a CKD user
    test_user = {
        "user_id":    1,
        "name":       "Ravi Kumar",
        "age":        45,
        "sex":        "male",
        "weight_kg":  72,
        "height_cm":  170,
        "activity":   "moderate",
        "condition":  "ckd"
    }

    # Simulate today's meals (ate Dal Makhani + Rajma)
    test_totals = {
        "meal_count":    2,
        "calories":      560,
        "protein_g":     33,    # close to CKD limit!
        "carbs_g":       66,
        "fat_g":         16,
        "potassium_mg":  1420,  # at 71% of CKD limit
        "phosphorus_mg": 530,   # at 66% of CKD limit
        "sodium_mg":     860,
        "fiber_g":       15.8,
    }

    print("=" * 55)
    print("   HEALTH RULE ENGINE — TEST RUN")
    print("=" * 55)

    result = check_meal_rules(test_user, test_totals)

    print(f"\n👤 User: {test_user['name']} | Condition: {test_user['condition'].upper()}")
    print(f"\n📊 DAILY LIMITS (personalised):")
    for k, v in result["limits"].items():
        print(f"   {k:20s}: {v}")

    print(f"\n📈 NUTRIENT STATUS:")
    for nutrient, data in result["percentages"].items():
        bar = "█" * int(data["percent"] / 10)
        print(f"   {data['label']:15s}: {data['consumed']:6.1f}/{data['limit']:6.1f}"
              f"{data['unit']:3s}  [{data['percent']:5.1f}%] {bar} [{data['level'].upper()}]")

    print(f"\n🚨 ALERTS ({len(result['alerts'])}):")
    for alert in result["alerts"]:
        print(f"   {alert['message']}")
        if alert["clinical_note"]:
            print(f"   📋 {alert['clinical_note']}")

    print(f"\n💬 SUMMARY: {result['summary']}")

    print(f"\n🥗 SAFE FOOD SUGGESTIONS:")
    for s in result["safe_swaps"]:
        print(f"   • {s}")

    # Test risk score
    avg = {
        "avg_daily_sodium": 1800,
        "avg_daily_protein": 55,
        "avg_daily_calories": 2100,
        "avg_daily_carbs": 200,
        "avg_daily_potassium": 2500,
    }
    risks = estimate_risk_score(avg, test_user)
    print(f"\n🔮 FUTURE RISK SCORES:")
    for k, v in risks.items():
        bar = "█" * (v // 10)
        print(f"   {k:20s}: {v:3d}/100  {bar}")