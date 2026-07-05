# 🥗 NutriCoach AI — Personalised Indian Food Nutrition Coach

An AI-powered nutrition coaching system that analyzes photos of Indian meals, tracks nutrient intake against clinical limits, and delivers personalised, hallucination-free coaching for patients managing Chronic Kidney Disease (CKD), Type 2 Diabetes, and Hypertension.

---

## 📌 Problem Statement

Patients with chronic conditions like CKD, diabetes, and hypertension must follow strict dietary limits — but:

- Generic calorie-tracking apps don't understand clinical nutrient limits (potassium, phosphorus, sodium)
- Most nutrition AI models are trained on Western food datasets and fail to recognize Indian dishes
- Generic LLM chatbots hallucinate medical/dietary facts, which is dangerous for clinical nutrition advice
- No existing tool combines food detection, clinical rule-checking, and evidence-grounded coaching in one place for Indian patients

**Research gap confirmed:** out of 2,053 studies reviewed on AI + nutrition, only 7 addressed AI-assisted dietary management for CKD patients specifically.

---

## ✅ What I Built & Achieved

- Built a **multi-food detection system** using YOLOv8 to identify every dish on an Indian thali from a single photo, achieving **82% mAP@50** across 30 Indian dish classes (trained on 5,446 images)
- Designed a **clinical rule engine** that checks daily nutrient intake (protein, potassium, phosphorus, sodium, calories, carbs) against condition-specific limits from **KDIGO 2024** and **ADA 2024** guidelines
- Personalised nutrient limits per user using the **Harris-Benedict equation** (TDEE) and body-weight-based protein scaling
- Built a **RAG pipeline** using ChromaDB and Sentence-Transformers (`all-MiniLM-L6-v2`) to ground every AI response in verified clinical knowledge, reducing hallucination rate from **38% to 6%**
- Integrated **Llama 3.3 70B (via Groq API)** to generate warm, doctor-style coaching responses referencing the patient's actual meals and nutrient status
- Implemented a **rule-based future risk score estimator** (0–100 scale) for CKD, diabetes, heart disease, and hypertension risk based on dietary patterns
- Designed a **fallback classification pipeline** (EfficientNetB3) for single-dish recognition when multi-food detection isn't available, plus a full **demo mode** that works without any trained model
- Built **portion-size estimation** directly from bounding box area, scaling nutrition values accordingly instead of assuming fixed serving sizes
- Deployed everything via **ONNX Runtime** for CPU-only inference — no GPU or TensorFlow dependency required

---

## 🧠 How It Works

1. **Upload a photo** of an Indian meal (thali, single dish, or snack)
2. **YOLOv8** detects every food item individually, drawing bounding boxes and estimating portion size from box area
3. **Nutrition lookup** matches each detected dish against a clinical nutrition database and scales values to the estimated portion
4. **Clinical rule engine** compares today's cumulative intake against the user's personalised daily limits and flags safe / warning / danger / exceeded levels
5. **RAG pipeline** retrieves the most relevant clinical guideline chunks from ChromaDB for the user's condition and question
6. **Llama 3.3 70B** generates a personalised coaching response — referencing the actual meal, exact nutrient numbers, and specific Indian food alternatives

---

## 📊 Key Results

| Metric | Result |
|---|---|
| YOLOv8 detection accuracy (mAP@50) | **82%** across 30 Indian dish classes |
| Training dataset size | 5,446 Indian food images |
| Hallucination rate — without RAG | 38% |
| Hallucination rate — with RAG | **6%** |
| Fallback model (EfficientNetB3) accuracy | 68.6% (89.75% top-5) |
| Detection confidence threshold | 0.15 (tunable) |

---

## 🛠️ Tools & Technologies

| Category | Tools Used |
|---|---|
| **Language** | Python |
| **Computer Vision** | YOLOv8 (ONNX), EfficientNetB3 (fallback), Pillow |
| **Model Inference** | ONNX Runtime (CPU) |
| **RAG / NLP** | ChromaDB, Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **LLM** | Llama 3.3 70B via Groq API |
| **Machine Learning** | Scikit-learn, XGBoost, Random Forest (risk scoring) |
| **Data Handling** | Pandas, NumPy |
| **Data Storage** | SQLite |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Web App** | Streamlit |

---

## 📂 Project Structure


---

## ⚙️ Installation Guide

### 1. Clone the repository
```bash
git clone https://github.com/Kalyan-Redd/Ncoach---AI.git
cd Ncoach---AI
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt**
### 4. Set up your Groq API key
```bash
export GROQ_API_KEY="your_gsk_key_here"   # On Windows: set GROQ_API_KEY=your_key
```
Get a free key at [console.groq.com](https://console.groq.com)

### 5. Run the application
```bash
streamlit run app.py
```

---

## 🚀 How to Use

1. **Upload a meal photo** — the system detects every food item and calculates nutrition
2. **Set up your profile** — age, weight, height, activity level, and health condition (CKD / Diabetes / Hypertension / None)
3. **View real-time clinical alerts** — track how close you are to daily nutrient limits
4. **Chat with your AI coach** — ask questions and get personalised, guideline-grounded advice
5. **Track weekly patterns** — view eating trends and future disease risk scores

*No trained model? The app runs in demo mode automatically using sample dishes.*

---

## 🔬 Research Foundation

- **KDIGO 2024 Clinical Practice Guidelines** — CKD nutrient limits
- **ADA 2024 Standards** — diabetes carbohydrate management
- **WHO Hypertension Guidelines 2023** — sodium/potassium recommendations
- **Nature Medicine, 2025 (Harvard, 105,000 participants, 30 years)** — diet quality and healthy aging
- **ScienceDirect Systematic Review, 2025** — AI + CKD nutrition research gap analysis

---

## 🔮 Future Improvements

- Replace rule-based risk scoring with a trained ML model on longitudinal patient data
- Add real-time barcode/packaged food scanning
- Expand the Indian food dataset beyond 30 dish classes
- Deploy on cloud with mobile app support
- Add multilingual coaching support (Hindi, Telugu, Tamil, etc.)

---

## 📫 Contact

**Kalyan Reddy Gaddam**
📧 kalyancoedu@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/gaddam-kalyan-reddy-482509283)
🤗 [HuggingFace](https://huggingface.co/coedu)
💻 [GitHub](https://github.com/Kalyan-Redd)

---

*This is an academic AI/ML research project. Not a substitute for professional medical advice — always consult a doctor or registered dietitian for personalised clinical guidance.*
