import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
from src.rag_pipeline import retrieve_context


# CONFIGURATION

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

 
# SYSTEM PROMPT


SYSTEM_PROMPT = SYSTEM_PROMPT = """
You are Dr. NutriCoach — an expert clinical nutritionist and AI health coach
specialising in Indian dietary management for patients with chronic conditions.

You have 15 years of experience managing CKD, Type 2 Diabetes, and Hypertension
patients on Indian diets. You speak like a warm, knowledgeable doctor who truly
cares about the patient's long-term health and longevity.

RESPONSE STRUCTURE — always follow this format:
1. Start by addressing the patient by name warmly
2. Comment specifically on what they ate today (mention the actual dish)
3. Give ONE clear main insight about their nutrition status
4. Give TWO or THREE specific actionable recommendations with exact Indian food names
5. End with a motivating sentence about their long-term health goal

TONE RULES:
- Sound like a real doctor talking to a patient — warm but authoritative
- Use "I recommend", "I suggest", "In my clinical experience"
- Be specific — say "Dal Makhani has 18g protein" not just "protein is high"
- Never be generic — always reference their actual meal and condition
- Mention specific Indian dishes by name — lauki sabzi, poha, idli, etc.
- Show you understand Indian culture — mention festivals, family meals, etc.

CLINICAL ACCURACY:
- ONLY use the verified clinical knowledge provided in the context
- Always cite why — "because CKD kidneys cannot filter excess potassium"
- Give exact numbers — "your potassium is at 71% of your 2000mg daily limit"
- Compare to their personal limits — not generic population averages

FORBIDDEN:
- Never give generic advice that ignores their actual meal data
- Never say "eat healthy" without specifying which Indian foods
- Never invent medical facts not in the provided knowledge context
- Never be preachy or repetitive
"""


# GROQ CLIENT
 

_client = None

def _get_client():
    """Initialise Groq client. Cached after first call."""
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def is_api_configured():
    """Check if Groq API key has been set."""
    return (
        GROQ_API_KEY != "PASTE_YOUR_GROQ_KEY_HERE" and
        GROQ_API_KEY.startswith("gsk_") and
        len(GROQ_API_KEY) > 20
    )


 
# PROMPT BUILDER


def build_coaching_prompt(user, today_totals, alerts,
                           rule_result, latest_meal=None,
                           weekly_summary=None, user_question=None):
    """Build rich multi-context prompt for the LLM."""

    condition = user.get("condition", "none")
    name      = user.get("name", "User")

    # RAG knowledge retrieval
    query       = user_question or f"nutrition advice for {condition} patient Indian diet"
    rag_context = retrieve_context(query, condition=condition, top_k=3)

    # Format alerts
    if not alerts:
        alert_text = "None — all nutrients within healthy limits today."
    else:
        alert_text = "\n".join(f"  • {a['message']}" for a in alerts)

    # Clinical notes
    clinical_notes = [a["clinical_note"] for a in alerts if a.get("clinical_note")]
    notes_text     = "\n".join(f"  {n}" for n in clinical_notes) if clinical_notes else "None."

    # Nutrient status
    pcts = rule_result.get("percentages", {})
    nutrient_lines = []
    for nutrient, data in pcts.items():
        label    = str(data.get("label", ""))
        consumed = data.get("consumed", 0)
        limit    = data.get("limit", 0)
        unit     = data.get("unit", "")
        percent  = data.get("percent", 0)
        level    = str(data.get("level", "")).upper()
        nutrient_lines.append(
            f"  {label:<15s}: {consumed}{unit} / {limit}{unit} "
            f"({percent}%) [{level}]"
        )
    nutrient_status = "\n".join(nutrient_lines)

    # Latest meal
    if latest_meal:
        meal_text = (
            f"  Dish       : {latest_meal.get('dish_name', 'Unknown')}\n"
            f"  Meal type  : {str(latest_meal.get('meal_type','meal')).title()}\n"
            f"  Calories   : {latest_meal.get('calories', 0)} kcal\n"
            f"  Protein    : {latest_meal.get('protein_g', 0)}g\n"
            f"  Potassium  : {latest_meal.get('potassium_mg', 0)}mg\n"
            f"  Phosphorus : {latest_meal.get('phosphorus_mg', 0)}mg\n"
            f"  Sodium     : {latest_meal.get('sodium_mg', 0)}mg"
        )
    else:
        meal_text = "No meal logged yet today."

    # Weekly summary
    def safe_float(val):
        try:
            return float(val or 0)
        except (TypeError, ValueError):
            return 0.0

    if weekly_summary:
        weekly_text = (
            f"  Avg daily calories  : {safe_float(weekly_summary.get('avg_daily_calories')):.0f} kcal\n"
            f"  Avg daily protein   : {safe_float(weekly_summary.get('avg_daily_protein')):.1f}g\n"
            f"  Avg daily sodium    : {safe_float(weekly_summary.get('avg_daily_sodium')):.0f}mg\n"
            f"  Avg daily potassium : {safe_float(weekly_summary.get('avg_daily_potassium')):.0f}mg\n"
            f"  Days tracked        : {weekly_summary.get('days_tracked') or 0}"
        )
    else:
        weekly_text = "No weekly data available yet."

    # Safe swaps
    swaps      = rule_result.get("safe_swaps", [])
    swaps_text = "\n".join(f"  • {s}" for s in swaps) if swaps else "  • Keep eating balanced meals."

    # Full prompt
    # Full prompt — structured for high quality response
    prompt = (
        f"You are coaching {name}, a {user.get('age','?')}-year-old "
        f"{str(user.get('sex','?'))} patient weighing {user.get('weight_kg','?')}kg "
        f"with {condition.upper().replace('_',' ')}.\n\n"

        f"WHAT THEY ATE TODAY:\n{meal_text}\n\n"

        f"THEIR NUTRIENT STATUS RIGHT NOW:\n{nutrient_status}\n\n"

        f"CLINICAL ALERTS TRIGGERED:\n{alert_text}\n\n"

        f"IMPORTANT CLINICAL NOTES FOR THIS CONDITION:\n{notes_text}\n\n"

        f"SAFER FOOD OPTIONS YOU SHOULD SUGGEST:\n{swaps_text}\n\n"

        f"THEIR WEEKLY EATING PATTERN:\n{weekly_text}\n\n"

        f"VERIFIED MEDICAL GUIDELINES TO USE:\n{rag_context}\n\n"

        f"THE PATIENT ASKS:\n"
        f"\"{user_question or 'What personalised advice do you have for me based on what I ate today?'}\"\n\n"

        f"INSTRUCTIONS FOR YOUR RESPONSE:\n"
        f"- Address {name} by name in your first sentence\n"
        f"- Mention their specific dish ({latest_meal.get('dish_name','their meal') if latest_meal else 'their meals'}) "
        f"and explain its nutritional impact for {condition} patients\n"
        f"- Give exact numbers from their nutrient status above\n"
        f"- Recommend 2-3 specific Indian dishes they should eat next\n"
        f"- Explain the long-term health consequence if they ignore this advice\n"
        f"- End with encouragement about living healthy past age 70\n"
        f"- Keep response between 150-250 words\n"
    )
    return prompt


# MAIN COACHING FUNCTION


def get_coaching_response(user, today_totals, alerts, rule_result,
                           latest_meal=None, weekly_summary=None,
                           user_question=None):
    """
    Generate personalised coaching response using Groq API.
    Falls back to rule-based demo response if API not configured.
    """
    if not is_api_configured():
        return _demo_response(user, alerts, rule_result)

    try:
        client = _get_client()
        prompt = build_coaching_prompt(
            user, today_totals, alerts, rule_result,
            latest_meal, weekly_summary, user_question
        )

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    except Exception as e:
        error_msg = str(e)
        print(f"Groq error: {error_msg}")

        if "auth" in error_msg.lower() or "api_key" in error_msg.lower():
            return (
                "⚠️ Groq API key invalid. Get a free key at console.groq.com\n\n"
                + _demo_response(user, alerts, rule_result)
            )
        if "rate" in error_msg.lower() or "429" in error_msg:
            return (
                "⚠️ Too many requests. Please wait 30 seconds and try again.\n\n"
                + _demo_response(user, alerts, rule_result)
            )
        return (
            f"⚠️ Coaching unavailable: {error_msg}\n\n"
            + _demo_response(user, alerts, rule_result)
        )



# DEMO FALLBACK RESPONSE


def _demo_response(user, alerts, rule_result):
    """Rule-based response when Groq API is not available."""
    name      = user.get("name", "there")
    condition = user.get("condition", "none").upper().replace("_", " ")
    summary   = rule_result.get("summary", "")
    swaps     = rule_result.get("safe_swaps", [])

    lines = [
        f"👋 Hello {name}! Here is your personalised nutrition coaching:\n",
        f"📊 **Today's Status:** {summary}\n",
    ]

    if alerts:
        lines.append("🚨 **Key Alerts:**")
        for a in alerts[:3]:
            lines.append(f"  • {a['message']}")
            if a.get("clinical_note"):
                lines.append(f"    ℹ️ {a['clinical_note']}")
        lines.append("")

    if swaps:
        lines.append("🥗 **Safer Food Options for Next Meal:**")
        for s in swaps:
            lines.append(f"  • {s}")
        lines.append("")

    lines.append(
        "💪 **Keep going!** Every healthy meal is a step toward living "
        "longer and feeling stronger. You are doing great!"
    )

    if condition != "NONE":
        lines.append(
            f"\n📋 *Advice tailored for {condition} patients. "
            "Always consult your doctor for personalised medical advice.*"
        )

    return "\n".join(lines)


 
# QUICK TEST — run: python src/nutrition_agent.py

if __name__ == "__main__":
    print("=" * 55)
    print("   NUTRITION AGENT — GROQ TEST")
    print("=" * 55)
    print(f"\nAPI configured: {is_api_configured()}")

    if is_api_configured():
        print("Testing Groq connection...")
        client = _get_client()
        r = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a nutrition coach."},
                {"role": "user",   "content": "Say hello in one sentence."}
            ],
            max_tokens=50,
        )
        print(f"Groq response: {r.choices[0].message.content}")
    else:
        print(" Paste your gsk_... key into GROQ_API_KEY variable.")