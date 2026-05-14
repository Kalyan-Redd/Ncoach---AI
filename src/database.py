
import sqlite3
import os
from datetime import datetime, date

# Database file lives in the project root
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ncoach.db")


def get_connection():
    """Get a SQLite connection. Creates DB file if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # allows dict-like access: row["column"]
    return conn


def create_tables():
    """
    Create all tables on first run.
    Safe to call every time — uses IF NOT EXISTS.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # ── USERS TABLE ──────────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            age         INTEGER NOT NULL,
            sex         TEXT NOT NULL CHECK(sex IN ('male','female')),
            weight_kg   REAL NOT NULL,
            height_cm   REAL NOT NULL,
            activity    TEXT NOT NULL DEFAULT 'moderate',
            condition   TEXT NOT NULL DEFAULT 'none',
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    # condition can be: none, ckd, diabetes, hypertension, ckd_diabetes

    # MEAL LOGS TABLE 
    # One row per meal photo uploaded
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS meal_logs (
            log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            log_date        TEXT NOT NULL,
            log_time        TEXT NOT NULL,
            meal_type       TEXT NOT NULL DEFAULT 'meal',
            dish_name       TEXT NOT NULL,
            confidence      REAL NOT NULL DEFAULT 0.0,
            calories        REAL NOT NULL DEFAULT 0.0,
            protein_g       REAL NOT NULL DEFAULT 0.0,
            carbs_g         REAL NOT NULL DEFAULT 0.0,
            fat_g           REAL NOT NULL DEFAULT 0.0,
            potassium_mg    REAL NOT NULL DEFAULT 0.0,
            phosphorus_mg   REAL NOT NULL DEFAULT 0.0,
            sodium_mg       REAL NOT NULL DEFAULT 0.0,
            fiber_g         REAL NOT NULL DEFAULT 0.0,
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)

    # ── RISK SCORES TABLE 
    # Stores ML-predicted future risk scores (updated weekly)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS risk_scores (
            score_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            score_date      TEXT NOT NULL,
            ckd_risk        REAL DEFAULT 0.0,
            diabetes_risk   REAL DEFAULT 0.0,
            heart_risk      REAL DEFAULT 0.0,
            hypert_risk     REAL DEFAULT 0.0,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)

    conn.commit()
    conn.close()
    print(" Database tables ready at:", DB_PATH)



# USER OPERATIONS


def create_user(name, age, sex, weight_kg, height_cm, activity, condition):
    """Insert a new user. Returns the new user_id."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO users (name, age, sex, weight_kg, height_cm, activity, condition)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (name, age, sex, weight_kg, height_cm, activity, condition))
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    return user_id


def get_user(user_id):
    """Get a single user by ID. Returns dict or None."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_users():
    """Return list of all users (for dropdown selection)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, name, condition FROM users ORDER BY name")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_user(user_id, **kwargs):
    """Update user fields dynamically. Pass keyword args for any field."""
    allowed = {"name","age","sex","weight_kg","height_cm","activity","condition"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return
    fields = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [user_id]
    conn = get_connection()
    conn.execute(f"UPDATE users SET {fields} WHERE user_id = ?", values)
    conn.commit()
    conn.close()



# MEAL LOG OPERATIONS


def log_meal(user_id, dish_name, confidence, nutrition, meal_type="meal"):
    """
    Store one meal in the database.

    Parameters:
        user_id    : int
        dish_name  : str  (e.g. "Dal Makhani")
        confidence : float (e.g. 0.92)
        nutrition  : dict with keys:
                     calories, protein_g, carbs_g, fat_g,
                     potassium_mg, phosphorus_mg, sodium_mg, fiber_g
        meal_type  : str  breakfast | lunch | dinner | snack | meal
    Returns:
        log_id (int)
    """
    now = datetime.now()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO meal_logs (
            user_id, log_date, log_time, meal_type,
            dish_name, confidence,
            calories, protein_g, carbs_g, fat_g,
            potassium_mg, phosphorus_mg, sodium_mg, fiber_g
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S"),
        meal_type,
        dish_name,
        confidence,
        nutrition.get("calories", 0),
        nutrition.get("protein_g", 0),
        nutrition.get("carbs_g", 0),
        nutrition.get("fat_g", 0),
        nutrition.get("potassium_mg", 0),
        nutrition.get("phosphorus_mg", 0),
        nutrition.get("sodium_mg", 0),
        nutrition.get("fiber_g", 0),
    ))
    conn.commit()
    log_id = cursor.lastrowid
    conn.close()
    return log_id


def get_today_meals(user_id):
    """Return all meals logged today for a user."""
    today = date.today().strftime("%Y-%m-%d")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM meal_logs
        WHERE user_id = ? AND log_date = ?
        ORDER BY log_time ASC
    """, (user_id, today))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_today_totals(user_id):
    """
    Sum all nutrients for today.
    Returns a single dict with totals. Returns zeros if no meals today.
    """
    today = date.today().strftime("%Y-%m-%d")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            COUNT(*)        AS meal_count,
            SUM(calories)   AS calories,
            SUM(protein_g)  AS protein_g,
            SUM(carbs_g)    AS carbs_g,
            SUM(fat_g)      AS fat_g,
            SUM(potassium_mg)  AS potassium_mg,
            SUM(phosphorus_mg) AS phosphorus_mg,
            SUM(sodium_mg)     AS sodium_mg,
            SUM(fiber_g)       AS fiber_g
        FROM meal_logs
        WHERE user_id = ? AND log_date = ?
    """, (user_id, today))
    row = cursor.fetchone()
    conn.close()
    if row and row["meal_count"] > 0:
        return dict(row)
    return {
        "meal_count": 0, "calories": 0, "protein_g": 0,
        "carbs_g": 0, "fat_g": 0, "potassium_mg": 0,
        "phosphorus_mg": 0, "sodium_mg": 0, "fiber_g": 0
    }


def get_weekly_meals(user_id, days=7):
    """
    Return all meals from the last N days.
    Used for weekly dashboard charts.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM meal_logs
        WHERE user_id = ?
          AND log_date >= date('now', ?)
        ORDER BY log_date ASC, log_time ASC
    """, (user_id, f"-{days} days"))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_weekly_daily_totals(user_id, days=7):
    """
    Return per-day nutrient totals for the last N days.
    Used for the weekly bar charts.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            log_date,
            COUNT(*)           AS meal_count,
            SUM(calories)      AS calories,
            SUM(protein_g)     AS protein_g,
            SUM(carbs_g)       AS carbs_g,
            SUM(fat_g)         AS fat_g,
            SUM(potassium_mg)  AS potassium_mg,
            SUM(phosphorus_mg) AS phosphorus_mg,
            SUM(sodium_mg)     AS sodium_mg,
            SUM(fiber_g)       AS fiber_g
        FROM meal_logs
        WHERE user_id = ?
          AND log_date >= date('now', ?)
        GROUP BY log_date
        ORDER BY log_date ASC
    """, (user_id, f"-{days} days"))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_most_eaten_dishes(user_id, days=30, limit=10):
    """Return top N most frequently eaten dishes in last N days."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT dish_name, COUNT(*) AS frequency,
               AVG(calories) AS avg_calories,
               AVG(protein_g) AS avg_protein
        FROM meal_logs
        WHERE user_id = ?
          AND log_date >= date('now', ?)
        GROUP BY dish_name
        ORDER BY frequency DESC
        LIMIT ?
    """, (user_id, f"-{days} days", limit))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_meals_for_risk(user_id, days=30):
    """
    Return feature-averaged data for ML risk prediction.
    Returns a dict of average daily nutrient values over last N days.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            AVG(daily_cal)   AS avg_daily_calories,
            AVG(daily_prot)  AS avg_daily_protein,
            AVG(daily_carbs) AS avg_daily_carbs,
            AVG(daily_fat)   AS avg_daily_fat,
            AVG(daily_k)     AS avg_daily_potassium,
            AVG(daily_p)     AS avg_daily_phosphorus,
            AVG(daily_na)    AS avg_daily_sodium,
            COUNT(*)         AS days_tracked
        FROM (
            SELECT
                log_date,
                SUM(calories)     AS daily_cal,
                SUM(protein_g)    AS daily_prot,
                SUM(carbs_g)      AS daily_carbs,
                SUM(fat_g)        AS daily_fat,
                SUM(potassium_mg) AS daily_k,
                SUM(phosphorus_mg)AS daily_p,
                SUM(sodium_mg)    AS daily_na
            FROM meal_logs
            WHERE user_id = ?
              AND log_date >= date('now', ?)
            GROUP BY log_date
        )
    """, (user_id, f"-{days} days"))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else {}


def delete_meal(log_id):
    """Delete a specific meal log entry."""
    conn = get_connection()
    conn.execute("DELETE FROM meal_logs WHERE log_id = ?", (log_id,))
    conn.commit()
    conn.close()



# RISK SCORE OPERATIONS


def save_risk_scores(user_id, ckd_risk, diabetes_risk, heart_risk, hypert_risk):
    """Save a new risk score entry for the user."""
    today = date.today().strftime("%Y-%m-%d")
    conn = get_connection()
    conn.execute("""
        INSERT INTO risk_scores (user_id, score_date, ckd_risk, diabetes_risk, heart_risk, hypert_risk)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, today, ckd_risk, diabetes_risk, heart_risk, hypert_risk))
    conn.commit()
    conn.close()


def get_latest_risk(user_id):
    """Get the most recent risk scores for a user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM risk_scores
        WHERE user_id = ?
        ORDER BY score_date DESC, score_id DESC
        LIMIT 1
    """, (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None



# INITIALIZE ON IMPORT

# Tables are created automatically whenever this module is imported
create_tables()



# QUICK TEST — run: python src/database.py

if __name__ == "__main__":
    # Create a test user
    uid = create_user(
        name="Ravi Kumar",
        age=35,
        sex="male",
        weight_kg=72,
        height_cm=170,
        activity="moderate",
        condition="ckd"
    )
    print(f" Test user created: user_id = {uid}")

    # Log a test meal
    lid = log_meal(
        user_id=uid,
        dish_name="Dal Makhani",
        confidence=0.92,
        nutrition={
            "calories": 330,
            "protein_g": 18,
            "carbs_g": 28,
            "fat_g": 14,
            "potassium_mg": 820,
            "phosphorus_mg": 290,
            "sodium_mg": 480,
            "fiber_g": 6
        },
        meal_type="lunch"
    )
    print(f" Meal logged: log_id = {lid}")

    # Get today totals
    totals = get_today_totals(uid)
    print(f" Today's totals: {totals}")

    # Get user
    user = get_user(uid)
    print(f" User fetched: {user}")