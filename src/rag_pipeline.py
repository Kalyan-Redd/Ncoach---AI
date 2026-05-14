import os
import chromadb
from chromadb.utils import embedding_functions


# PATHS


BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
DATA_DIR   = os.path.join(BASE_DIR, "data")

COLLECTION_NAME = "nutrition_knowledge"
EMBED_MODEL     = "all-MiniLM-L6-v2"   # lightweight, fast, accurate



# CLINICAL KNOWLEDGE BASE

# Each entry is a knowledge chunk that will be embedded and stored.
# Written as factual, clinical statements for precise retrieval.

KNOWLEDGE_BASE = [

    # ── CKD Guidelines 
    {
        "id":       "ckd_protein_1",
        "text":     "Chronic Kidney Disease (CKD) patients should limit daily protein "
                    "intake to 0.6–0.8 grams per kilogram of body weight. Excess protein "
                    "increases nitrogenous waste that damaged kidneys cannot filter, "
                    "accelerating disease progression. Source: KDIGO 2024 Guidelines.",
        "topic":    "ckd_protein",
        "condition":"ckd",
    },
    {
        "id":       "ckd_potassium_1",
        "text":     "CKD patients must restrict potassium intake to less than 2000mg per day. "
                    "Damaged kidneys cannot excrete potassium properly, leading to hyperkalemia "
                    "which can cause life-threatening cardiac arrhythmias. High-potassium Indian "
                    "foods to avoid include banana, rajma, dal, spinach, and coconut water.",
        "topic":    "ckd_potassium",
        "condition":"ckd",
    },
    {
        "id":       "ckd_phosphorus_1",
        "text":     "Phosphorus should be limited to 800mg per day for CKD patients. "
                    "High phosphorus weakens bones (renal osteodystrophy) and causes "
                    "calcification of blood vessels. Indian foods high in phosphorus include "
                    "paneer, milk, curd, nuts, and whole grains.",
        "topic":    "ckd_phosphorus",
        "condition":"ckd",
    },
    {
        "id":       "ckd_sodium_1",
        "text":     "CKD patients should restrict sodium to below 1500mg per day. "
                    "Excess sodium causes fluid retention, hypertension, and increases "
                    "the workload on the kidneys. Indian foods very high in sodium include "
                    "pickle (achar), papad, namkeen, packaged snacks, and restaurant curries.",
        "topic":    "ckd_sodium",
        "condition":"ckd",
    },
    {
        "id":       "ckd_safe_foods_1",
        "text":     "Safe Indian foods for CKD patients include: lauki (bottle gourd) sabzi, "
                    "white rice, cabbage sabzi, arbi (colocasia), white bread, apple, "
                    "cucumber, and egg white curry. These are low in potassium, phosphorus, "
                    "and protein making them kidney-friendly choices.",
        "topic":    "ckd_safe_foods",
        "condition":"ckd",
    },
    {
        "id":       "ckd_fluids_1",
        "text":     "CKD patients in later stages (Stage 3b-5) may need to restrict fluid "
                    "intake to 1–1.5 litres per day including all beverages and water content "
                    "in foods. Fluid restriction prevents oedema, breathlessness, and "
                    "hypertension associated with fluid overload.",
        "topic":    "ckd_fluids",
        "condition":"ckd",
    },

    # ── Diabetes Guidelines 
    {
        "id":       "diabetes_carbs_1",
        "text":     "Type 2 diabetes patients should limit carbohydrate intake to 130–225g "
                    "per day depending on body weight and insulin sensitivity. "
                    "Consistent carbohydrate distribution across meals helps maintain "
                    "stable blood glucose levels. Choose complex carbohydrates like "
                    "whole grains over refined carbohydrates like maida. Source: ADA 2024.",
        "topic":    "diabetes_carbs",
        "condition":"diabetes",
    },
    {
        "id":       "diabetes_fiber_1",
        "text":     "Diabetes patients should consume at least 30g of dietary fiber per day. "
                    "Fiber slows glucose absorption, improves insulin sensitivity, and "
                    "reduces post-meal blood sugar spikes. Good fiber sources in Indian diet "
                    "include rajma, moong dal, vegetables, and fruits with skin.",
        "topic":    "diabetes_fiber",
        "condition":"diabetes",
    },
    {
        "id":       "diabetes_safe_foods_1",
        "text":     "Diabetes-friendly Indian foods include: methi (fenugreek) dishes, "
                    "karela (bitter gourd) sabzi, moong dal, palak sabzi, tandoori chicken, "
                    "egg bhurji, dosa (in moderation), and curd. These foods have a lower "
                    "glycaemic index and help maintain blood glucose control.",
        "topic":    "diabetes_safe_foods",
        "condition":"diabetes",
    },
    {
        "id":       "diabetes_avoid_1",
        "text":     "Diabetes patients should avoid or strictly limit: white rice in large "
                    "portions, maida-based foods like naan and bhature, sugary sweets like "
                    "gulab jamun and jalebi, sweetened lassi, fruit juices, and fried snacks "
                    "like samosa and pakora which cause rapid blood sugar spikes.",
        "topic":    "diabetes_avoid",
        "condition":"diabetes",
    },

    # ── Hypertension Guidelines ───────────────────────────────────────────────
    {
        "id":       "hypert_sodium_1",
        "text":     "Hypertension patients must restrict sodium to less than 1500mg per day. "
                    "The DASH diet recommends reducing sodium as the single most effective "
                    "dietary intervention for lowering blood pressure. Each 1000mg reduction "
                    "in sodium can lower systolic blood pressure by 5–6 mmHg. "
                    "Source: WHO Hypertension Guidelines 2023.",
        "topic":    "hypert_sodium",
        "condition":"hypertension",
    },
    {
        "id":       "hypert_potassium_1",
        "text":     "Hypertension patients should increase potassium intake to 3500mg per day. "
                    "Potassium counteracts the blood-pressure-raising effects of sodium. "
                    "Good potassium sources in the Indian diet include banana, rajma, "
                    "spinach, and sweet potato — unless the patient also has CKD.",
        "topic":    "hypert_potassium",
        "condition":"hypertension",
    },

    # ── General Nutrition & Longevity 
    {
        "id":       "longevity_diet_1",
        "text":     "A Harvard study of 105,000 adults over 30 years found that participants "
                    "with the highest quality diets had an 86% greater likelihood of healthy "
                    "aging at age 70 — defined as freedom from chronic disease with preserved "
                    "physical and cognitive function. Diet quality is the single most "
                    "actionable lever for longevity. Source: Nature Medicine, 2025.",
        "topic":    "longevity",
        "condition":"general",
    },
    {
        "id":       "longevity_processed_1",
        "text":     "Diets high in ultra-processed foods are directly linked to premature "
                    "mortality, cognitive decline, and chronic disease. Ultra-processed foods "
                    "include packaged snacks, instant noodles, processed meats, sweetened "
                    "beverages, and fast food. Replacing one ultra-processed serving daily "
                    "with whole food reduces mortality risk by 8–19%.",
        "topic":    "longevity_processed",
        "condition":"general",
    },
    {
        "id":       "general_calories_1",
        "text":     "Daily calorie requirements are calculated using the Harris-Benedict "
                    "equation. A moderately active adult male requires approximately 2200–2500 "
                    "kcal/day while a moderately active female requires 1800–2100 kcal/day. "
                    "Consistently eating 500 kcal above TDEE leads to approximately 0.5kg "
                    "weight gain per week.",
        "topic":    "calories",
        "condition":"general",
    },
    {
        "id":       "indian_diet_1",
        "text":     "Traditional Indian diet when balanced is inherently healthy — rich in "
                    "legumes (dal, rajma, chana) for plant protein and fiber, fermented foods "
                    "(idli, dosa, curd) for probiotics, and spices (turmeric, cumin, coriander) "
                    "with anti-inflammatory properties. Problems arise with excessive oil, "
                    "salt, refined carbohydrates, and portion sizes.",
        "topic":    "indian_diet",
        "condition":"general",
    },
    {
        "id":       "indian_diet_2",
        "text":     "Indian street food and restaurant food is typically very high in sodium "
                    "(1500–2500mg per serving), saturated fat, and refined carbohydrates. "
                    "Home-cooked Indian food with controlled oil and salt is significantly "
                    "healthier. Patients with chronic conditions should prefer home-cooked "
                    "meals and avoid restaurant food as much as possible.",
        "topic":    "indian_restaurant",
        "condition":"general",
    },
    {
        "id":       "meal_timing_1",
        "text":     "For all chronic conditions, meal timing and frequency matter. "
                    "Eating 3 balanced meals at consistent times prevents blood sugar spikes, "
                    "reduces overeating, and supports kidney function. Skipping meals leads "
                    "to compensatory overeating. Breakfast should not be skipped as it "
                    "stabilises morning blood glucose and prevents mid-day energy crashes.",
        "topic":    "meal_timing",
        "condition":"general",
    },
    {
        "id":       "weight_1",
        "text":     "Maintaining a healthy BMI (18.5–24.9) significantly reduces risk of "
                    "all chronic conditions. Research shows even a 5kg weight gain during "
                    "young adulthood increases Type 2 diabetes risk by 30%, hypertension by "
                    "14%, and premature mortality by 5%. Weight management through diet "
                    "is more effective than exercise alone. Source: Journal of Internal Medicine 2024.",
        "topic":    "weight",
        "condition":"general",
    },
]



# CHROMA CLIENT & COLLECTION


_chroma_client     = None
_chroma_collection = None


def _get_collection():
    """
    Initialise ChromaDB client and collection.
    Creates the collection and embeds knowledge base on first run.
    Returns the existing collection on subsequent calls.
    """
    global _chroma_client, _chroma_collection

    if _chroma_collection is not None:
        return _chroma_collection

    # Persistent client — saves to disk in chroma_db/ folder
    _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Sentence-transformer embedding function
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    # Get or create collection
    _chroma_collection = _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}   # cosine similarity
    )

    # Populate if empty
    if _chroma_collection.count() == 0:
        print("📚 Building knowledge base — this runs only once...")
        _populate_knowledge_base(_chroma_collection)
        print(f"✅ Knowledge base ready: {_chroma_collection.count()} chunks indexed.")
    else:
        print(f"✅ Knowledge base loaded: {_chroma_collection.count()} chunks.")

    return _chroma_collection


def _populate_knowledge_base(collection):
    """
    Embed all knowledge chunks and insert into ChromaDB.
    Called once on first run.
    """
    ids       = [chunk["id"]   for chunk in KNOWLEDGE_BASE]
    documents = [chunk["text"] for chunk in KNOWLEDGE_BASE]
    metadatas = [
        {
            "topic":     chunk["topic"],
            "condition": chunk["condition"],
        }
        for chunk in KNOWLEDGE_BASE
    ]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )



# RETRIEVAL FUNCTION


def retrieve_context(query, condition="none", top_k=3):
    """
    Retrieve the most relevant knowledge chunks for a given query.

    Uses cosine similarity between query embedding and stored chunk
    embeddings to find the most relevant clinical guidelines.

    Parameters:
        query     : str  — user's question or coaching context
        condition : str  — user's health condition for filtering
        top_k     : int  — number of chunks to retrieve (default 3)

    Returns:
        str — concatenated relevant knowledge chunks,
              ready to be injected into the LLM prompt.
    """
    collection = _get_collection()

    # Build where filter to prioritise condition-specific chunks
    # Also include general chunks always
    if condition and condition != "none":
        where_filter = {
            "$or": [
                {"condition": {"$eq": condition}},
                {"condition": {"$eq": "general"}},
            ]
        }
    else:
        where_filter = None

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
            where=where_filter,
            include=["documents", "distances", "metadatas"]
        )

        chunks    = results["documents"][0]
        distances = results["distances"][0]

        if not chunks:
            return _fallback_context(condition)

        # Format context with relevance indicator
        context_parts = []
        for i, (chunk, dist) in enumerate(zip(chunks, distances)):
            relevance = round((1 - dist) * 100, 1)   # cosine → percentage
            context_parts.append(
                f"[Knowledge {i+1} | Relevance: {relevance}%]\n{chunk}"
            )

        return "\n\n".join(context_parts)

    except Exception as e:
        print(f"⚠️ RAG retrieval error: {e}")
        return _fallback_context(condition)


def _fallback_context(condition):
    """Return basic fallback context if ChromaDB query fails."""
    return (
        "General nutrition guidance: Eat balanced meals with adequate vegetables, "
        "whole grains, and lean protein. Avoid processed and high-sodium foods. "
        "Consult your doctor or registered dietitian for personalised advice."
    )



# KNOWLEDGE BASE MANAGEMENT


def rebuild_knowledge_base():
    """
    Force rebuild of the ChromaDB knowledge base.
    Call this if you update the KNOWLEDGE_BASE list.
    """
    global _chroma_collection
    collection = _get_collection()

    if collection.count() > 0:
        # Delete all existing documents
        all_ids = [chunk["id"] for chunk in KNOWLEDGE_BASE]
        try:
            collection.delete(ids=all_ids)
        except Exception:
            pass

    _populate_knowledge_base(collection)
    _chroma_collection = collection
    print(f"✅ Knowledge base rebuilt: {collection.count()} chunks.")


def get_knowledge_stats():
    """Return stats about the current knowledge base."""
    collection = _get_collection()
    conditions = {}
    for chunk in KNOWLEDGE_BASE:
        c = chunk["condition"]
        conditions[c] = conditions.get(c, 0) + 1
    return {
        "total_chunks": collection.count(),
        "by_condition": conditions,
    }



# QUICK TEST — run: python src/rag_pipeline.py

if __name__ == "__main__":
    print("=" * 55)
    print("   RAG PIPELINE — TEST RUN")
    print("=" * 55)

    stats = get_knowledge_stats()
    print(f"\n📚 Knowledge Base Stats:")
    print(f"   Total chunks : {stats['total_chunks']}")
    for cond, count in stats["by_condition"].items():
        print(f"   {cond:15s} : {count} chunks")

    test_queries = [
        ("What foods should a CKD patient avoid?",        "ckd"),
        ("How much protein can I eat with kidney disease?","ckd"),
        ("Which Indian foods are safe for diabetes?",     "diabetes"),
        ("How does diet affect longevity?",               "general"),
    ]

    print(f"\n🔍 Retrieval Tests:\n")
    for query, condition in test_queries:
        print(f"Q: {query}")
        print(f"   Condition: {condition}")
        context = retrieve_context(query, condition=condition, top_k=2)
        # Show only first 200 chars for readability
        preview = context[:200].replace("\n", " ") + "..."
        print(f"   Context  : {preview}")
        print()