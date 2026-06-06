"""Streamlit interface for YOLOv10 agricultural field analysis."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision.ops import nms
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from torchvision.transforms import Normalize
from ultralytics import YOLO
from ultralytics.engine.results import Boxes


MODEL_PATH = Path(__file__).with_name("best.pt")
FIELD_MODEL_PATH = Path(__file__).with_name("field_boundary_model.pt")
FIELD_BOUNDS_DIR = Path(__file__).parent / "data" / "raw" / "data2017_miniscale" / "field_bounds"
CLASS_LABELS = {
    "planter_skip": "Eksik ekim",
    "double_plant": "Cift veya sik ekim",
    "weed_cluster": "Duzensiz bitki ortusu (yabanci ot suphesi)",
    "drydown": "Kuruma suphesi",
    "water": "Su birikmesi",
}


@st.cache_resource
def load_model() -> YOLO:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model dosyasi bulunamadi: {MODEL_PATH}")
    return YOLO(str(MODEL_PATH))


@st.cache_resource
def load_field_model():
    if not FIELD_MODEL_PATH.exists():
        return None

    model = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None)
    model.classifier.low_classifier = nn.Conv2d(40, 1, kernel_size=1)
    model.classifier.high_classifier = nn.Conv2d(128, 1, kernel_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(FIELD_MODEL_PATH, map_location=device, weights_only=True))
    model.to(device).eval()
    return model, device


def predict_field_mask(image_array: np.ndarray) -> np.ndarray | None:
    loaded = load_field_model()
    if loaded is None:
        return None

    model, device = loaded
    image = torch.from_numpy(image_array.astype(np.float32) / 255.0).permute(2, 0, 1)
    image = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(image)
    with torch.no_grad():
        prediction = model(image.unsqueeze(0).to(device))["out"].sigmoid()[0, 0]
    return prediction.cpu().numpy() > 0.5


def load_field_mask(
    filename: str, image_array: np.ndarray
) -> tuple[np.ndarray | None, str | None]:
    predicted_mask = predict_field_mask(image_array)
    if predicted_mask is not None:
        return predicted_mask, "model"

    mask_path = FIELD_BOUNDS_DIR / f"{Path(filename).stem}.png"
    if not mask_path.exists():
        return None, None

    mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None, None
    if mask.shape != image_array.shape[:2]:
        mask = cv2.resize(
            mask, (image_array.shape[1], image_array.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return mask > 0, "dataset"


def calculate_report(
    result, image_shape: tuple[int, int], field_mask: np.ndarray | None = None
) -> list[dict[str, object]]:
    image_height, image_width = image_shape
    image_area = int(field_mask.sum()) if field_mask is not None else image_height * image_width
    boxes = result.boxes
    rows: list[dict[str, object]] = []
    if boxes is None or len(boxes) == 0:
        return rows

    names = result.names
    counts = Counter(int(class_id) for class_id in boxes.cls.cpu().tolist())
    areas: Counter[int] = Counter()
    for class_id, coordinates in zip(boxes.cls.cpu().tolist(), boxes.xyxy.cpu().tolist()):
        x1, y1, x2, y2 = [int(value) for value in coordinates]
        if field_mask is None:
            areas[int(class_id)] += max(0, x2 - x1) * max(0, y2 - y1)
        else:
            areas[int(class_id)] += int(field_mask[y1:y2, x1:x2].sum())

    for class_id, count in sorted(counts.items()):
        class_name = names[class_id]
        rows.append(
            {
                "Sorun": CLASS_LABELS.get(class_name, class_name),
                "Tespit sayisi": count,
                "Yaklasik alan orani": f"%{100 * areas[class_id] / image_area:.2f}",
            }
        )
    return rows


def suppress_overlapping_boxes(result, iou_threshold: float = 0.35):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return result

    kept_indices: list[int] = []
    for class_id in boxes.cls.unique():
        class_indices = (boxes.cls == class_id).nonzero(as_tuple=True)[0]
        selected = nms(boxes.xyxy[class_indices], boxes.conf[class_indices], iou_threshold)
        kept_indices.extend(class_indices[selected].cpu().tolist())

    kept_indices.sort()
    result.boxes = Boxes(boxes.data[kept_indices], result.orig_shape)
    return result


def filter_boxes_by_field(result, field_mask: np.ndarray | None, min_field_ratio: float = 0.5):
    boxes = result.boxes
    if field_mask is None or boxes is None or len(boxes) == 0:
        return result

    kept_indices: list[int] = []
    for index, coordinates in enumerate(boxes.xyxy.cpu().tolist()):
        x1, y1, x2, y2 = [int(value) for value in coordinates]
        box_area = max(0, x2 - x1) * max(0, y2 - y1)
        if box_area == 0:
            continue
        field_area = int(field_mask[y1:y2, x1:x2].sum())
        if field_area / box_area >= min_field_ratio:
            kept_indices.append(index)

    result.boxes = Boxes(boxes.data[kept_indices], result.orig_shape)
    return result


def create_field_overlay(image_array: np.ndarray, field_mask: np.ndarray) -> np.ndarray:
    overlay = image_array.copy()
    outside_field = ~field_mask
    overlay[outside_field] = (0.35 * overlay[outside_field]).astype(np.uint8)
    contours, _ = cv2.findContours(
        field_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return cv2.drawContours(overlay, contours, -1, (255, 215, 0), 2)


def main() -> None:
    st.set_page_config(page_title="Drone Tarim Arazisi Analizi", layout="wide")
    st.title("Drone ile Tarim Arazisi Analizi")
    st.write(
        "Drone goruntusunu yukleyin. YOLOv10 modeli eksik ekim, sik ekim, "
        "duzensiz bitki ortusu ve kuruma suphelerini isaretlesin."
    )

    confidence = st.sidebar.slider("Guven esigi", 0.01, 0.90, 0.20, 0.01)
    st.sidebar.caption(
        "Dengeli analiz icin 0.20 ile baslayin. Daha fazla supheli alan gormek "
        "icin esigi azaltin; yalnizca guclu tahminler icin artirin."
    )
    uploaded_file = st.file_uploader("Arazi goruntusu secin", type=("jpg", "jpeg", "png"))
    if uploaded_file is None:
        st.info("Analiz icin bir drone veya hava goruntusu yukleyin.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.asarray(image)
    field_mask, field_mask_source = load_field_mask(uploaded_file.name, image_array)
    model = load_model()

    with st.spinner("Arazi analiz ediliyor..."):
        prediction_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        result = model.predict(prediction_image, conf=confidence, verbose=False)[0]
        result = suppress_overlapping_boxes(result)
        result = filter_boxes_by_field(result, field_mask)

    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    report = calculate_report(result, image_array.shape[:2], field_mask)

    original_column, result_column = st.columns(2)
    with original_column:
        st.subheader("Orijinal goruntu")
        st.image(image_array, use_container_width=True)
    with result_column:
        st.subheader("YOLOv10 analiz sonucu")
        st.image(annotated_rgb, use_container_width=True)

    st.subheader("Arazi raporu")
    if field_mask_source == "model":
        st.info("Tarim arazisi siniri otomatik tahmin edildi. Arazi disindaki tespitler filtrelendi.")
    elif field_mask_source == "dataset":
        st.info("Veri seti sinir maskesi bulundu. Arazi disindaki tespitler filtrelendi.")
    else:
        st.warning(
            "Bu goruntu icin arazi siniri maskesi bulunamadi. "
            "Tespitler tum goruntu uzerinden hesaplandi."
        )
    if field_mask is not None:
        with st.expander("Tarim arazisi sinirini goster"):
            st.caption(
                "Sari cizginin ici analiz edilen tarim alanidir. "
                "Karartilan bolgeler sorun raporuna dahil edilmez."
            )
            st.image(create_field_overlay(image_array, field_mask), use_container_width=True)
    if report:
        total_detections = sum(int(row["Tespit sayisi"]) for row in report)
        st.metric("Toplam sorunlu bolge", total_detections)
        st.dataframe(report, use_container_width=True, hide_index=True)
    else:
        st.success("Secilen guven esiginde sorunlu bolge tespit edilmedi.")


if __name__ == "__main__":
    main()
