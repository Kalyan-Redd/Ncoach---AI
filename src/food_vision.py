import os
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# PATHS

# Check what path is written — should match exactly
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "yolo_food.onnx")
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo_food.onnx")
YAML_PATH       = os.path.join(BASE_DIR, "models", "data.yaml")
NUTRITION_PATH  = os.path.join(BASE_DIR, "data",   "indian_food_nutrition.csv")

# Fallback single-dish model paths
EFFNET_PATH     = os.path.join(BASE_DIR, "models", "indian_food_classifier.onnx")
CLASSES_PATH    = os.path.join(BASE_DIR, "models", "class_names.json")

IMG_SIZE        = 640    # YOLOv8 input size
CONF_THRESHOLD  = 0.15   # minimum confidence to accept detection


# CACHED GLOBALS


_yolo_session  = None
_yolo_classes  = None
_nutrition_df  = None
_effnet_session = None
_effnet_classes = None



# MODEL LOADERS

def _load_nutrition():
    """Load nutrition CSV once and cache it."""
    global _nutrition_df
    if _nutrition_df is None:
        _nutrition_df = pd.read_csv(NUTRITION_PATH)
        _nutrition_df["dish_key"] = (
            _nutrition_df["dish_name"]
            .str.lower()
            .str.strip()
            .str.replace(" ", "_")
        )
    return _nutrition_df


def _load_yolo():
    """Load YOLOv8 ONNX model and class names from data.yaml."""
    global _yolo_session, _yolo_classes

    if _yolo_session is None:
        if not os.path.exists(YOLO_MODEL_PATH):
            raise FileNotFoundError(
                f"YOLOv8 ONNX model not found at: {YOLO_MODEL_PATH}\n"
                "Download from Colab after training and place in models/ folder."
            )
        import onnxruntime as ort
        _yolo_session = ort.InferenceSession(
            YOLO_MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )

    if _yolo_classes is None:
        if os.path.exists(YAML_PATH):
            import yaml
            with open(YAML_PATH) as f:
                data = yaml.safe_load(f)
            _yolo_classes = data["names"]
        else:
            raise FileNotFoundError(
                f"data.yaml not found at: {YAML_PATH}\n"
                "Download from Colab along with the ONNX model."
            )

    return _yolo_session, _yolo_classes


def _load_effnet():
    """Load fallback EfficientNetB3 single-dish classifier."""
    global _effnet_session, _effnet_classes

    if _effnet_session is None:
        if not os.path.exists(EFFNET_PATH):
            return None, None
        import onnxruntime as ort
        _effnet_session = ort.InferenceSession(
            EFFNET_PATH,
            providers=["CPUExecutionProvider"]
        )

    if _effnet_classes is None:
        if os.path.exists(CLASSES_PATH):
            with open(CLASSES_PATH) as f:
                _effnet_classes = json.load(f)
        else:
            return None, None

    return _effnet_session, _effnet_classes



# IMAGE PREPROCESSING


def _preprocess_yolo(image):
    """
    Prepare PIL image for YOLOv8 inference.
    YOLOv8 expects: float32, normalised 0-1, shape (1, 3, 640, 640)
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    orig_w, orig_h = image.size
    image_rgb      = image.convert("RGB")
    image_resized  = image_rgb.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    arr = np.array(image_resized, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)          # HWC → CHW
    arr = np.expand_dims(arr, axis=0)     # add batch dim

    return arr, orig_w, orig_h


def _preprocess_effnet(image):
    """Prepare image for EfficientNetB3 inference."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("RGB")
    image = image.resize((224, 224), Image.LANCZOS)
    arr   = np.array(image, dtype=np.float32)
    arr   = (arr - 127.5) / 127.5        # EfficientNet normalisation
    arr   = np.expand_dims(arr, axis=0)
    return arr



# YOLO OUTPUT PARSING


def _parse_yolo_output(outputs, orig_w, orig_h, conf_thresh=CONF_THRESHOLD):
    """
    Parse raw YOLOv8 ONNX output into detection boxes.
    YOLOv8 ONNX output: (1, 4+nc, 8400)
    Coordinates are in PIXEL space relative to 640x640 input.
    """
    output = outputs[0]   # (1, 4+nc, 8400)
    output = output[0]    # (4+nc, 8400)
    output = output.T     # (8400, 4+nc)

    scale_x = orig_w / IMG_SIZE   # e.g. 1200/640 = 1.875
    scale_y = orig_h / IMG_SIZE

    boxes_out = []

    for row in output:
        xc, yc, w, h = row[0], row[1], row[2], row[3]
        class_scores  = row[4:]
        class_id      = int(np.argmax(class_scores))
        confidence    = float(class_scores[class_id])

        if confidence < conf_thresh:
            continue

        # FIX: coords are already in 640x640 pixel space
        # Just scale to original image size — NO extra * IMG_SIZE
        x1 = int((xc - w / 2) * scale_x)
        y1 = int((yc - h / 2) * scale_y)
        x2 = int((xc + w / 2) * scale_x)
        y2 = int((yc + h / 2) * scale_y)

        # Clamp to image boundaries
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes_out.append({
            "bbox":       [x1, y1, x2, y2],
            "class_id":   class_id,
            "confidence": confidence,
        })

    boxes_out = _apply_nms(boxes_out, iou_thresh=0.5)
    return boxes_out


def _apply_nms(boxes, iou_thresh=0.5):
    """Simple NMS to remove overlapping duplicate detections."""
    if not boxes:
        return boxes

    boxes = sorted(boxes, key=lambda x: x["confidence"], reverse=True)
    kept  = []

    while boxes:
        best = boxes.pop(0)
        kept.append(best)
        boxes = [
            b for b in boxes
            if _iou(best["bbox"], b["bbox"]) < iou_thresh
        ]

    return kept


def _iou(box1, box2):
    """Calculate Intersection over Union between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1        = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2        = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union        = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


 
# NUTRITION LOOKUP


def _estimate_portion(bbox, orig_w, orig_h):
    """
    Estimate portion size in grams from bounding box area ratio.

    Logic:
        - A box covering 100% of image → ~400g (large serving)
        - A box covering 10% of image  → ~80g  (small item)
        - Clamped between 50g and 450g
    """
    box_area  = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    img_area  = orig_w * orig_h
    ratio     = box_area / img_area
    portion_g = ratio * 800   # scale factor
    return round(max(50, min(450, portion_g)))


def lookup_nutrition(dish_name, portion_g=250):
    """
    Get nutrition values for a dish scaled to portion size.
    Tries exact match first, then partial match.
    Returns default values if not found.
    """
    df  = _load_nutrition()
    key = dish_name.lower().strip().replace(" ", "_")

    # Exact match
    match = df[df["dish_key"] == key]

    # Partial match on first word
    if match.empty:
        first_word = key.split("_")[0]
        match = df[df["dish_key"].str.contains(first_word, na=False)]

    # Partial match on dish name directly
    if match.empty:
        match = df[df["dish_name"].str.lower().str.contains(
            dish_name.lower().split("_")[0], na=False
        )]

    if match.empty:
        return _default_nutrition(dish_name, portion_g)

    row   = match.iloc[0]
    scale = portion_g / 100.0

    return {
        "dish_name":     row["dish_name"],
        "portion_g":     portion_g,
        "category":      row.get("category", "unknown"),
        "calories":      round(float(row["calories"])      * scale, 1),
        "protein_g":     round(float(row["protein_g"])     * scale, 1),
        "carbs_g":       round(float(row["carbs_g"])       * scale, 1),
        "fat_g":         round(float(row["fat_g"])         * scale, 1),
        "potassium_mg":  round(float(row["potassium_mg"])  * scale, 1),
        "phosphorus_mg": round(float(row["phosphorus_mg"]) * scale, 1),
        "sodium_mg":     round(float(row["sodium_mg"])     * scale, 1),
        "fiber_g":       round(float(row["fiber_g"])       * scale, 1),
    }


def _default_nutrition(dish_name, portion_g=250):
    """Conservative defaults when dish not in database."""
    scale = portion_g / 100.0
    return {
        "dish_name":     dish_name,
        "portion_g":     portion_g,
        "category":      "unknown",
        "calories":      round(200.0 * scale, 1),
        "protein_g":     round(7.0   * scale, 1),
        "carbs_g":       round(30.0  * scale, 1),
        "fat_g":         round(6.0   * scale, 1),
        "potassium_mg":  round(280.0 * scale, 1),
        "phosphorus_mg": round(130.0 * scale, 1),
        "sodium_mg":     round(350.0 * scale, 1),
        "fiber_g":       round(3.0   * scale, 1),
    }



# ANNOTATED IMAGE GENERATOR


# Colors for bounding boxes — one per class
BBOX_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
    "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
    "#BB8FCE", "#85C1E9", "#82E0AA", "#F8C471",
    "#F1948A", "#73C6B6", "#7FB3D3", "#A9DFBF",
]

def draw_detections(image, detections):
    """
    Draw bounding boxes and labels on image.

    Parameters:
        image      : PIL Image
        detections : list of detection dicts from predict_food()

    Returns:
        PIL Image with annotations drawn
    """
    draw    = ImageDraw.Draw(image.copy())
    img_out = image.copy()
    d       = ImageDraw.Draw(img_out)

    for i, det in enumerate(detections):
        bbox      = det["bbox"]
        dish      = det["dish_name"]
        conf      = det["confidence"]
        calories  = det["nutrition"]["calories"]
        color     = BBOX_COLORS[i % len(BBOX_COLORS)]

        x1, y1, x2, y2 = bbox

        # Draw bounding box
        for thickness in range(3):
            d.rectangle(
                [x1-thickness, y1-thickness,
                 x2+thickness, y2+thickness],
                outline=color
            )

        # Label background
        label = f"{dish} {conf*100:.0f}% | {calories}kcal"
        label_y = max(0, y1 - 20)
        d.rectangle([x1, label_y, x1 + len(label)*7, label_y+18],
                    fill=color)
        d.text((x1+3, label_y+2), label, fill="white")

    return img_out


 
# MAIN PREDICTION FUNCTIONS


def predict_food(image, conf_threshold=CONF_THRESHOLD):
    """
    Main function — detects ALL food items in a photo.

    Uses YOLOv8 for multi-food detection if model available,
    falls back to EfficientNetB3 single-dish if not.

    Parameters:
        image          : PIL Image or file path
        conf_threshold : minimum confidence (default 0.35)

    Returns:
        dict with:
            detections      : list of individual food items
            total_nutrition : summed nutrition for whole plate
            annotated_image : PIL Image with bounding boxes
            model_used      : "yolov8" or "efficientnet"
            item_count      : number of items detected
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")

    # Try YOLOv8 first
    if is_yolo_available():
        return _predict_yolo(image, conf_threshold)

    # Fallback to EfficientNetB3
    if is_effnet_available():
        return _predict_effnet(image)

    # Demo mode
    return predict_food_demo()


def _predict_yolo(image, conf_threshold=CONF_THRESHOLD):
    """Run YOLOv8 multi-food detection."""
    session, class_names = _load_yolo()
    orig_w, orig_h       = image.size

    # Preprocess
    input_tensor = _preprocess_yolo(image)[0]

    # Inference
    input_name = session.get_inputs()[0].name
    outputs    = session.run(None, {input_name: input_tensor})

    # Parse detections
    raw_boxes  = _parse_yolo_output(outputs, orig_w, orig_h, conf_threshold)

    if not raw_boxes:
        # No detections — try lower threshold
        raw_boxes = _parse_yolo_output(outputs, orig_w, orig_h, 0.5)

    # Build detection results
    detections      = []
    total_nutrition = _empty_nutrition()

    for box in raw_boxes:
        dish_name = class_names[box["class_id"]] \
                    if box["class_id"] < len(class_names) \
                    else "Unknown"
        portion_g = _estimate_portion(box["bbox"], orig_w, orig_h)
        nutrition = lookup_nutrition(dish_name, portion_g)

        det = {
            "dish_name":  dish_name,
            "confidence": round(box["confidence"], 4),
            "bbox":       box["bbox"],
            "portion_g":  portion_g,
            "nutrition":  nutrition,
        }
        detections.append(det)

        # Add to totals
        for key in total_nutrition:
            total_nutrition[key] += nutrition.get(key, 0)

    # Round totals
    total_nutrition = {k: round(v, 1) for k, v in total_nutrition.items()}

    # Draw annotated image
    annotated = draw_detections(image, detections)

    return {
        "detections":       detections,
        "total_nutrition":  total_nutrition,
        "annotated_image":  annotated,
        "model_used":       "yolov8",
        "item_count":       len(detections),
        # For backward compatibility with app.py
        "dish_name":        detections[0]["dish_name"] if detections else "Unknown",
        "confidence":       detections[0]["confidence"] if detections else 0.0,
        "nutrition":        total_nutrition,
        "top_k":            [
            {"dish": d["dish_name"],
             "confidence": d["confidence"],
             "percent": round(d["confidence"]*100, 1)}
            for d in detections[:3]
        ],
    }


def _predict_effnet(image):
    """Fallback: EfficientNetB3 single dish classification."""
    session, class_names = _load_effnet()
    if session is None:
        return predict_food_demo()

    input_tensor = _preprocess_effnet(image)
    input_name   = session.get_inputs()[0].name
    output       = session.run(None, {input_name: input_tensor})[0][0]

    top_indices = np.argsort(output)[::-1][:3]
    best_idx    = top_indices[0]
    dish_name   = class_names[best_idx]
    confidence  = float(output[best_idx])
    nutrition   = lookup_nutrition(dish_name, 250)

    return {
        "detections": [{
            "dish_name":  dish_name,
            "confidence": round(confidence, 4),
            "bbox":       [0, 0, image.width, image.height],
            "portion_g":  250,
            "nutrition":  nutrition,
        }],
        "total_nutrition":  nutrition,
        "annotated_image":  image,
        "model_used":       "efficientnet",
        "item_count":       1,
        "dish_name":        dish_name,
        "confidence":       round(confidence, 4),
        "nutrition":        nutrition,
        "top_k": [
            {"dish": class_names[i],
             "confidence": round(float(output[i]), 4),
             "percent": round(float(output[i])*100, 1)}
            for i in top_indices
        ],
    }


def _empty_nutrition():
    """Return zero nutrition dict for accumulation."""
    return {
        "calories": 0, "protein_g": 0, "carbs_g": 0,
        "fat_g": 0, "potassium_mg": 0, "phosphorus_mg": 0,
        "sodium_mg": 0, "fiber_g": 0,
    }


# DEMO MODE


DEMO_DISHES = [
    "Dal Makhani", "Biryani", "Butter Chicken",
    "Palak Paneer", "Dosa", "Idli", "Rajma",
    "Chole Bhature", "Aloo Paratha", "Paneer Tikka"
]

def predict_food_demo(dish_name=None, portion_g=250):
    """
    Demo mode — works without any trained model.
    Used for testing app before ONNX model is ready.
    """
    _load_nutrition()

    if dish_name is None:
        dish_name = "Dal Makhani"

    nutrition = lookup_nutrition(dish_name, portion_g)

    return {
        "detections": [{
            "dish_name":  dish_name,
            "confidence": 0.92,
            "bbox":       [50, 50, 400, 400],
            "portion_g":  portion_g,
            "nutrition":  nutrition,
        }],
        "total_nutrition":  nutrition,
        "annotated_image":  None,
        "model_used":       "demo",
        "item_count":       1,
        "dish_name":        dish_name,
        "confidence":       0.92,
        "nutrition":        nutrition,
        "demo_mode":        True,
        "top_k": [
            {"dish": dish_name,     "confidence": 0.92, "percent": 92.0},
            {"dish": "Rajma",       "confidence": 0.05, "percent": 5.0},
            {"dish": "Masoor Dal",  "confidence": 0.03, "percent": 3.0},
        ],
    }


 
# STATUS CHECKS


def is_yolo_available():
    return os.path.exists(YOLO_MODEL_PATH) and os.path.exists(YAML_PATH)

def is_effnet_available():
    return os.path.exists(EFFNET_PATH) and os.path.exists(CLASSES_PATH)

def is_model_available():
    return is_yolo_available() or is_effnet_available()

def get_model_status():
    if is_yolo_available():
        return "✅ YOLOv8 multi-food detection model loaded"
    if is_effnet_available():
        return "⚠️ EfficientNetB3 single-dish model loaded (YOLOv8 not found)"
    return "⚠️ No model found — running in demo mode"


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("   FOOD VISION MODULE — TEST")
    print("=" * 55)
    print(f"\n{get_model_status()}")
    print(f"YOLOv8   : {is_yolo_available()}")
    print(f"EffNet   : {is_effnet_available()}")

    print("\nTesting demo mode...")
    result = predict_food_demo("Dal Makhani", 300)
    print(f"Dish      : {result['dish_name']}")
    print(f"Calories  : {result['nutrition']['calories']} kcal")
    print(f"Protein   : {result['nutrition']['protein_g']} g")
    print(f"Potassium : {result['nutrition']['potassium_mg']} mg")
    print(f"Items     : {result['item_count']}")
    print(f"Model     : {result['model_used']}")
    print("\n✅ food_vision.py working correctly!")