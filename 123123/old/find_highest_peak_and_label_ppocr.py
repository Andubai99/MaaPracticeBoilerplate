import os
import re
import cv2
import numpy as np
import onnxruntime as ort
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt

# =========================
# 可改参数
# =========================
IMAGE_PATH = "p1.png"  # 改成你的截图路径
SHOW_FIGURE = True
REC_MODEL_PATH = r"F:\123123\ocr\rec.onnx"
KEYS_PATH = r"F:\123123\ocr\keys.txt"


class PPOCRRecognizer:
    def __init__(self, model_path, keys_path, providers=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        # 常见为 [1,3,48,320] 或动态宽度
        self.input_h = int(input_shape[2]) if isinstance(input_shape[2], int) else 48
        self.input_w = int(input_shape[3]) if isinstance(input_shape[3], int) else 320
        self.characters = self._load_keys(keys_path)
        self.blank_idx = 0

    @staticmethod
    def _load_keys(keys_path):
        chars = []
        with open(keys_path, "r", encoding="utf-8") as f:
            for line in f:
                ch = line.rstrip("\n\r")
                if ch:
                    chars.append(ch)
        return ["blank"] + chars

    def preprocess(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        ratio = w / max(h, 1)
        resized_w = int(np.ceil(self.input_h * ratio))
        resized_w = max(8, min(resized_w, self.input_w))
        resized = cv2.resize(img, (resized_w, self.input_h), interpolation=cv2.INTER_CUBIC)
        resized = resized.astype(np.float32) / 255.0
        resized = (resized - 0.5) / 0.5
        padded = np.zeros((self.input_h, self.input_w, 3), dtype=np.float32)
        padded[:, :resized_w, :] = resized
        chw = np.transpose(padded, (2, 0, 1))[None, :, :, :]
        return chw

    def decode(self, logits):
        if logits.ndim == 3:
            preds_idx = np.argmax(logits, axis=2)[0]
            preds_prob = np.max(logits, axis=2)[0]
        elif logits.ndim == 2:
            preds_idx = np.argmax(logits, axis=1)
            preds_prob = np.max(logits, axis=1)
        else:
            raise RuntimeError(f"Unexpected OCR output shape: {logits.shape}")

        char_list = []
        conf_list = []
        last_idx = None
        for idx, prob in zip(preds_idx, preds_prob):
            idx = int(idx)
            if idx == self.blank_idx or idx == last_idx:
                last_idx = idx
                continue
            if 0 <= idx < len(self.characters):
                char_list.append(self.characters[idx])
                conf_list.append(float(prob))
            last_idx = idx
        text = "".join(char_list)
        score = float(np.mean(conf_list)) if conf_list else 0.0
        return text, score

    def recognize(self, img_bgr):
        inp = self.preprocess(img_bgr)
        outputs = self.session.run(None, {self.input_name: inp})
        text, score = self.decode(outputs[0])
        return text, score


def merge_boxes(boxes, x_gap=20, min_y_overlap=60):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = [list(boxes[0][:4])]
    for b in boxes[1:]:
        x1, y1, x2, y2, *_ = b
        lx1, ly1, lx2, ly2 = merged[-1]
        y_overlap = min(ly2, y2) - max(ly1, y1)
        if x1 <= lx2 + x_gap and y_overlap > min_y_overlap:
            merged[-1] = [min(lx1, x1), min(ly1, y1), max(lx2, x2), max(ly2, y2)]
        else:
            merged.append([x1, y1, x2, y2])
    return merged


def extract_curve_top_y(roi_mask):
    cols = np.where(roi_mask.sum(axis=0) > 0)[0]
    if len(cols) == 0:
        return None, None

    xs, ys = [], []
    for c in cols:
        rows = np.where(roi_mask[:, c] > 0)[0]
        if len(rows) == 0:
            continue
        xs.append(c)
        ys.append(rows.min())

    if not xs:
        return None, None

    return np.array(xs), np.array(ys, dtype=float)


def detect_plot_boxes(blue_mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blue_mask, 8)
    boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > 500:
            boxes.append([x, y, x + w, y + h, area])

    plot_boxes = merge_boxes(boxes, x_gap=20, min_y_overlap=60)
    if not plot_boxes:
        raise RuntimeError("没有检测到谱图区域，请检查截图质量或蓝色阈值。")
    return plot_boxes


def find_highest_peak_in_plot(blue_mask, box):
    x1, y1, x2, y2 = box
    roi_mask = blue_mask[y1:y2, x1:x2]
    xs, ys = extract_curve_top_y(roi_mask)
    if xs is None:
        raise RuntimeError("当前谱图区域没有提取到蓝色曲线。")

    full_x = np.arange(xs.min(), xs.max() + 1)
    full_y = np.interp(full_x, xs, ys)

    if len(full_y) >= 11:
        window = 11
    else:
        window = max(5, (len(full_y) // 2) * 2 + 1)
    if window >= len(full_y):
        window = len(full_y) - 1 if len(full_y) % 2 == 0 else len(full_y)
    if window < 5:
        smooth_y = full_y.copy()
    else:
        smooth_y = savgol_filter(full_y, window_length=window, polyorder=2)

    signal = -smooth_y
    peaks, _ = find_peaks(signal, prominence=3, distance=15)

    if len(peaks) == 0:
        best_i = int(np.argmin(smooth_y))
    else:
        best_i = peaks[np.argmax(signal[peaks])]

    peak_x = int(full_x[best_i] + x1)
    peak_y = float(smooth_y[best_i] + y1)
    return peak_x, peak_y


def preprocess_for_text_detection(crop_bgr):
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([90, 40, 40]), np.array([145, 255, 255]))
    clean = crop_bgr.copy()
    clean[blue_mask > 0] = (255, 255, 255)
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 去掉细小噪点，稍微横向连通字符
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    merged = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel_close)
    return clean, merged


def parse_float_like(text):
    text = text.strip().replace(",", "")
    text = text.lstrip("*")
    m = re.search(r"\d+\.\d+|\d+", text)
    if not m:
        return None
    try:
        return float(m.group())
    except Exception:
        return None


def text_boxes_from_crop(crop_bgr, crop_origin_xy, recognizer):
    clean, merged = preprocess_for_text_detection(crop_bgr)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(merged, 8)
    ox, oy = crop_origin_xy
    candidates = []

    H, W = merged.shape[:2]
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 20:
            continue
        if w < 8 or h < 6:
            continue
        if h > H * 0.7 or w > W * 0.95:
            continue

        pad = 3
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h + pad)
        text_crop = clean[y0:y1, x0:x1]
        text, score = recognizer.recognize(text_crop)
        text = text.strip()
        val = parse_float_like(text)
        if val is None:
            continue

        left = ox + x0
        top = oy + y0
        bw = x1 - x0
        bh = y1 - y0
        cx = left + bw / 2
        cy = top + bh / 2
        candidates.append({
            "text": text,
            "value": val,
            "conf": score * 100.0,
            "bbox": (int(left), int(top), int(bw), int(bh)),
            "center": (cx, cy),
        })

    return candidates


def choose_best_label(candidates, peak_x, peak_y):
    if not candidates:
        return None

    best = None
    best_score = 1e18
    for c in candidates:
        cx, cy = c["center"]
        dx = abs(cx - peak_x)
        dy = abs(cy - peak_y)
        below_penalty = 120 if cy > peak_y + 35 else 0
        conf_penalty = max(0.0, 95.0 - c["conf"]) * 0.5
        score = dx * 1.0 + dy * 1.2 + below_penalty + conf_penalty
        if score < best_score:
            best_score = score
            best = c
    return best


def read_peak_label(img_bgr, plot_box, peak_x, peak_y, recognizer):
    x1, y1, x2, y2 = plot_box
    h, w = img_bgr.shape[:2]

    left = max(x1, peak_x - 130)
    right = min(x2, peak_x + 180)
    top = max(0, int(peak_y) - 110)
    bottom = min(h, int(peak_y) + 60)

    local_crop = img_bgr[top:bottom, left:right]
    local_candidates = text_boxes_from_crop(local_crop, (left, top), recognizer)
    best = choose_best_label(local_candidates, peak_x, peak_y)
    if best is not None:
        return best

    plot_crop = img_bgr[y1:y2, x1:x2]
    plot_candidates = text_boxes_from_crop(plot_crop, (x1, y1), recognizer)
    best = choose_best_label(plot_candidates, peak_x, peak_y)
    return best


def annotate_and_save(img_bgr, results, save_path):
    annotated = img_bgr.copy()
    for item in results:
        x1, y1, x2, y2 = item["roi"]
        peak_x, peak_y = item["peak_pixel"]
        label = item.get("label_text")
        value = item.get("label_value")

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.circle(annotated, (peak_x, int(round(peak_y))), 8, (0, 0, 255), 2)

        text = f"Peak {item['plot_index']}"
        if value is not None:
            text += f": {value}"
        elif label:
            text += f": {label}"

        text_y = max(25, int(round(peak_y)) - 12)
        cv2.putText(
            annotated,
            text,
            (peak_x + 10, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        if item.get("label_bbox") is not None:
            lx, ly, lw, lh = item["label_bbox"]
            cv2.rectangle(annotated, (lx, ly), (lx + lw, ly + lh), (255, 0, 255), 1)

    cv2.imwrite(save_path, annotated)
    return annotated


def main(image_path=IMAGE_PATH, show_figure=SHOW_FIGURE):
    if not os.path.isfile(REC_MODEL_PATH):
        raise FileNotFoundError(f"找不到识别模型：{REC_MODEL_PATH}")
    if not os.path.isfile(KEYS_PATH):
        raise FileNotFoundError(f"找不到字符字典：{KEYS_PATH}")

    recognizer = PPOCRRecognizer(REC_MODEL_PATH, KEYS_PATH)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"读不到图片：{image_path}")

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([140, 255, 255]))
    plot_boxes = detect_plot_boxes(blue_mask)

    results = []
    for idx, box in enumerate(plot_boxes, start=1):
        peak_x, peak_y = find_highest_peak_in_plot(blue_mask, box)
        label_info = read_peak_label(img_bgr, box, peak_x, peak_y, recognizer)

        item = {
            "plot_index": idx,
            "roi": tuple(map(int, box)),
            "peak_pixel": (int(peak_x), round(float(peak_y), 2)),
            "label_text": None,
            "label_value": None,
            "label_bbox": None,
        }

        if label_info is not None:
            item["label_text"] = label_info["text"]
            item["label_value"] = label_info["value"]
            item["label_bbox"] = label_info["bbox"]

        results.append(item)

    base, _ = os.path.splitext(image_path)
    save_img = base + "_annotated.png"
    save_txt = base + "_results.txt"

    annotated = annotate_and_save(img_bgr, results, save_img)

    with open(save_txt, "w", encoding="utf-8") as f:
        f.write("检测结果\n")
        f.write("=" * 50 + "\n")
        for r in results:
            f.write(f"图 {r['plot_index']}\n")
            f.write(f"  谱图区域 ROI: {r['roi']}\n")
            f.write(f"  最高峰像素: {r['peak_pixel']}\n")
            f.write(f"  读出的峰旁质量值文本: {r['label_text']}\n")
            f.write(f"  解析后的质量值: {r['label_value']}\n")
            f.write(f"  标签框: {r['label_bbox']}\n")
            f.write("-" * 50 + "\n")

    print("\n检测结果：")
    for r in results:
        print(f"图 {r['plot_index']}: 最高峰像素 = {r['peak_pixel']}, 质量值 = {r['label_value']}, 原始文本 = {r['label_text']}")
    print(f"\n已保存标注图：{save_img}")
    print(f"已保存结果文本：{save_txt}")

    if show_figure:
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(16, 5))
        plt.imshow(annotated_rgb)
        plt.axis("off")
        plt.title("Highest peak + label value (PP-OCR ONNX)")
        plt.tight_layout()
        plt.show()

    return results


if __name__ == "__main__":
    main()
