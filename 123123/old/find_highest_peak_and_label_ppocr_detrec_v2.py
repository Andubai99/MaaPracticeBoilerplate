import os
import re
import cv2
import numpy as np
import onnxruntime as ort
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt

IMAGE_PATH = "p1.jpg"
SHOW_FIGURE = True
DET_MODEL_PATH = r"F:\123123\ocr\det.onnx"
REC_MODEL_PATH = r"F:\123123\ocr\rec.onnx"
KEYS_PATH = r"F:\123123\ocr\keys.txt"

FULL_MASS_REGEX = re.compile(r"\d{4}\.\d{4}")
PARTIAL_NUM_REGEX = re.compile(r"\d+(?:\.\d+)?")


class PPOCRDetector:
    def __init__(self, model_path, providers=None, max_side_len=960, thresh=0.25, box_thresh=0.35, unclip_ratio=1.8):
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.max_side_len = max_side_len
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio

    def preprocess(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = min(1.0, float(self.max_side_len) / max(h, w))
        resize_h = max(32, int(round(h * scale / 32) * 32))
        resize_w = max(32, int(round(w * scale / 32) * 32))
        resized = cv2.resize(img, (resize_w, resize_h))
        x = resized.astype(np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = np.transpose(x, (2, 0, 1))[None, :, :, :]
        return x, (h, w)

    @staticmethod
    def order_points_clockwise(pts):
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @staticmethod
    def get_mini_boxes(contour):
        bounding_box = cv2.minAreaRect(contour)
        points = cv2.boxPoints(bounding_box)
        points = np.array(points, dtype=np.float32)
        points = PPOCRDetector.order_points_clockwise(points)
        side = min(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[1] - points[2]))
        return points, side

    @staticmethod
    def box_score_fast(bitmap, box):
        h, w = bitmap.shape[:2]
        box = box.copy().astype(np.int32)
        box[:, 0] = np.clip(box[:, 0], 0, w - 1)
        box[:, 1] = np.clip(box[:, 1], 0, h - 1)
        x_min = np.min(box[:, 0])
        x_max = np.max(box[:, 0])
        y_min = np.min(box[:, 1])
        y_max = np.max(box[:, 1])
        mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
        box_local = box.copy()
        box_local[:, 0] -= x_min
        box_local[:, 1] -= y_min
        cv2.fillPoly(mask, [box_local], 1)
        return cv2.mean(bitmap[y_min:y_max + 1, x_min:x_max + 1], mask)[0]

    def unclip(self, box):
        center = np.mean(box, axis=0, keepdims=True)
        return (box - center) * self.unclip_ratio + center

    def boxes_from_bitmap(self, pred, bitmap, dest_w, dest_h):
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            if contour.shape[0] < 4:
                continue
            box, sside = self.get_mini_boxes(contour)
            if sside < 3:
                continue
            score = self.box_score_fast(pred, box)
            if score < self.box_thresh:
                continue
            box = self.unclip(box)
            box, sside = self.get_mini_boxes(box.reshape(-1, 1, 2).astype(np.float32))
            if sside < 3:
                continue
            box[:, 0] = np.clip(np.round(box[:, 0] / pred.shape[1] * dest_w), 0, dest_w - 1)
            box[:, 1] = np.clip(np.round(box[:, 1] / pred.shape[0] * dest_h), 0, dest_h - 1)
            boxes.append(box.astype(np.int32))
        return boxes

    def detect(self, img_bgr):
        inp, (orig_h, orig_w) = self.preprocess(img_bgr)
        outputs = self.session.run(None, {self.input_name: inp})
        pred = outputs[0]
        if pred.ndim == 4:
            pred = pred[0, 0]
        elif pred.ndim == 3:
            pred = pred[0]
        else:
            raise RuntimeError(f"Unexpected det output shape: {outputs[0].shape}")
        bitmap = (pred > self.thresh).astype(np.uint8)
        return self.boxes_from_bitmap(pred, bitmap, orig_w, orig_h)


class PPOCRRecognizer:
    def __init__(self, model_path, keys_path, providers=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
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
        chars, probs, last_idx = [], [], None
        for idx, prob in zip(preds_idx, preds_prob):
            idx = int(idx)
            if idx == self.blank_idx or idx == last_idx:
                last_idx = idx
                continue
            if 0 <= idx < len(self.characters):
                chars.append(self.characters[idx])
                probs.append(float(prob))
            last_idx = idx
        return "".join(chars), (float(np.mean(probs)) if probs else 0.0)

    def recognize(self, img_bgr):
        inp = self.preprocess(img_bgr)
        outputs = self.session.run(None, {self.input_name: inp})
        return self.decode(outputs[0])


def normalize_numeric_text(text):
    if text is None:
        return ""
    text = text.strip()
    text = text.replace(",", ".")
    text = text.replace(" ", "")
    replacements = {
        "O": "0", "o": "0", "D": "0", "Q": "0",
        "I": "1", "l": "1", "|": "1",
        "S": "5", "s": "5",
        "B": "8",
        "—": "", "-": "", "_": "", "*": ""
    }
    text = "".join(replacements.get(ch, ch) for ch in text)
    text = re.sub(r"[^0-9.]", "", text)
    return text


def parse_mass_from_text(text):
    s = normalize_numeric_text(text)
    m = FULL_MASS_REGEX.search(s)
    if m:
        full = m.group()
        return full, float(full)
    digits = re.sub(r"\D", "", s)
    if len(digits) >= 8:
        full = digits[:4] + "." + digits[4:8]
        return full, float(full)
    m2 = PARTIAL_NUM_REGEX.search(s)
    if m2:
        token = m2.group()
        try:
            return token, float(token)
        except Exception:
            pass
    return None, None


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
    window = 11 if len(full_y) >= 11 else max(5, (len(full_y) // 2) * 2 + 1)
    if window >= len(full_y):
        window = len(full_y) - 1 if len(full_y) % 2 == 0 else len(full_y)
    smooth_y = full_y.copy() if window < 5 else savgol_filter(full_y, window_length=window, polyorder=2)
    signal = -smooth_y
    peaks, _ = find_peaks(signal, prominence=3, distance=15)
    best_i = int(np.argmin(smooth_y)) if len(peaks) == 0 else peaks[np.argmax(signal[peaks])]
    return int(full_x[best_i] + x1), float(smooth_y[best_i] + y1)


def preprocess_local_for_text(local_crop):
    hsv = cv2.cvtColor(local_crop, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([90, 40, 40]), np.array([145, 255, 255]))
    clean = local_crop.copy()
    clean[blue_mask > 0] = (255, 255, 255)
    return clean


def get_rotate_crop_image(img, points):
    pts = np.array(points, dtype=np.float32)
    width = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    height = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    width = max(width, 8)
    height = max(height, 8)
    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (width, height), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if warped.shape[0] > warped.shape[1] * 1.5:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


def box_center(box):
    arr = np.array(box, dtype=np.float32)
    return float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))


def recognize_multi(recognizer, crop_bgr):
    variants = []
    base = crop_bgr.copy()
    variants.append(base)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(th1, cv2.COLOR_GRAY2BGR))
    _, th2 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    variants.append(cv2.cvtColor(th2, cv2.COLOR_GRAY2BGR))
    big = cv2.resize(base, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    variants.append(big)

    best = ("", 0.0)
    for v in variants:
        text, score = recognizer.recognize(v)
        norm, value = parse_mass_from_text(text)
        if norm and FULL_MASS_REGEX.fullmatch(norm):
            return norm, value, score
        if score > best[1]:
            best = (text, score)
    norm, value = parse_mass_from_text(best[0])
    return norm if norm else best[0], value, best[1]


def components_in_strip(strip_bgr):
    gray = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    # 先轻微膨胀，让小数点和相邻数字更容易并到一起
    dil = cv2.dilate(inv, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dil, 8)
    boxes = []
    H, W = inv.shape[:2]
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 6 or w < 2 or h < 5:
            continue
        if h > H * 0.95 or w > W * 0.98:
            continue
        boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[0])
    return boxes


def group_boxes_into_lines(boxes, y_tol=8, x_gap=10):
    groups = []
    for b in boxes:
        x, y, w, h = b
        cy = y + h / 2.0
        placed = False
        for g in groups:
            gcy = np.mean([bb[1] + bb[3] / 2.0 for bb in g])
            gxmax = max(bb[0] + bb[2] for bb in g)
            if abs(cy - gcy) <= y_tol and x <= gxmax + x_gap:
                g.append(b)
                placed = True
                break
        if not placed:
            groups.append([b])
    return groups


def read_full_mass_near_peak(img_bgr, plot_box, peak_x, peak_y, recognizer):
    x1, y1, x2, y2 = plot_box
    H, W = img_bgr.shape[:2]
    # 标签一般在峰顶上方偏右
    left = max(x1, peak_x - 12)
    right = min(x2, peak_x + 155)
    top = max(0, int(peak_y) - 28)
    bottom = min(H, int(peak_y) + 6)
    strip = preprocess_local_for_text(img_bgr[top:bottom, left:right])
    boxes = components_in_strip(strip)
    if not boxes:
        return None
    groups = group_boxes_into_lines(boxes, y_tol=6, x_gap=8)

    candidates = []
    for g in groups:
        gx0 = min(bb[0] for bb in g)
        gy0 = min(bb[1] for bb in g)
        gx1 = max(bb[0] + bb[2] for bb in g)
        gy1 = max(bb[1] + bb[3] for bb in g)
        # 扩大右边界，确保把小数点和后四位一起吃进去
        pad_l, pad_r, pad_t, pad_b = 4, 18, 4, 4
        x0 = max(0, gx0 - pad_l)
        y0 = max(0, gy0 - pad_t)
        x1c = min(strip.shape[1], gx1 + pad_r)
        y1c = min(strip.shape[0], gy1 + pad_b)
        crop = strip[y0:y1c, x0:x1c]
        text, value, score = recognize_multi(recognizer, crop)
        norm, val2 = parse_mass_from_text(text)
        value = val2 if val2 is not None else value
        global_bbox = (left + x0, top + y0, x1c - x0, y1c - y0)
        center = (global_bbox[0] + global_bbox[2] / 2.0, global_bbox[1] + global_bbox[3] / 2.0)
        if norm is None and value is None:
            continue
        candidates.append({
            "text": norm if norm else text,
            "value": value,
            "score": score,
            "bbox": global_bbox,
            "center": center,
        })

    if not candidates:
        return None

    def score_candidate(c):
        cx, cy = c["center"]
        bbox = c["bbox"]
        dx = abs(cx - (peak_x + 35))
        dy = abs(cy - (peak_y - 8))
        right_penalty = 0 if cx >= peak_x - 5 else 120
        short_penalty = 0 if FULL_MASS_REGEX.fullmatch(str(c["text"])) else 80
        width_penalty = 0 if bbox[2] >= 48 else 40
        return dx + 1.5 * dy + right_penalty + short_penalty + width_penalty - 20 * c["score"]

    candidates.sort(key=score_candidate)
    return candidates[0]


def read_peak_label(img_bgr, plot_box, peak_x, peak_y, detector, recognizer):
    # 先用“固定格式 4+4”专门读取最高峰旁边整段文本
    full = read_full_mass_near_peak(img_bgr, plot_box, peak_x, peak_y, recognizer)
    if full is not None and full.get("text"):
        norm, value = parse_mass_from_text(full["text"])
        full["text"] = norm if norm else full["text"]
        full["value"] = value
        if FULL_MASS_REGEX.fullmatch(str(full["text"])):
            return full

    # 再走 det+rec 通用链路
    x1, y1, x2, y2 = plot_box
    H, W = img_bgr.shape[:2]
    left = max(x1, peak_x - 180)
    right = min(x2, peak_x + 220)
    top = max(0, int(peak_y) - 130)
    bottom = min(H, int(peak_y) + 80)
    local_crop = img_bgr[top:bottom, left:right]
    local_clean = preprocess_local_for_text(local_crop)

    candidates = []
    try:
        det_boxes = detector.detect(local_clean)
        for box in det_boxes:
            crop = get_rotate_crop_image(local_clean, box)
            text, value, score = recognize_multi(recognizer, crop)
            norm, val2 = parse_mass_from_text(text)
            value = val2 if val2 is not None else value
            if norm is None and value is None:
                continue
            gbox = box.copy()
            gbox[:, 0] += left
            gbox[:, 1] += top
            gx_min, gy_min = int(gbox[:, 0].min()), int(gbox[:, 1].min())
            gx_max, gy_max = int(gbox[:, 0].max()), int(gbox[:, 1].max())
            candidates.append({
                "text": norm if norm else text,
                "value": value,
                "score": score,
                "bbox": (gx_min, gy_min, gx_max - gx_min, gy_max - gy_min),
                "center": box_center(gbox),
            })
    except Exception:
        pass

    # 回退：若 det 框太碎，使用专门 strip 结果
    if full is not None:
        candidates.append(full)

    if not candidates:
        return None

    def generic_score(c):
        cx, cy = c["center"]
        dx = abs(cx - peak_x)
        dy = abs(cy - peak_y)
        below_penalty = 140 if cy > peak_y + 40 else 0
        short_penalty = 0 if FULL_MASS_REGEX.fullmatch(str(c["text"])) else 80
        return dx + 1.3 * dy + below_penalty + short_penalty - 15 * c["score"]

    best = min(candidates, key=generic_score)
    norm, value = parse_mass_from_text(best["text"])
    best["text"] = norm if norm else best["text"]
    best["value"] = value
    return best


def annotate_and_save(img_bgr, results, save_path):
    annotated = img_bgr.copy()
    for item in results:
        x1, y1, x2, y2 = item["roi"]
        peak_x, peak_y = item["peak_pixel"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.circle(annotated, (peak_x, int(round(peak_y))), 8, (0, 0, 255), 2)
        text = f"Peak {item['plot_index']}"
        if item.get("label_text"):
            text += f": {item['label_text']}"
        elif item.get("label_value") is not None:
            text += f": {item['label_value']}"
        text_y = max(25, int(round(peak_y)) - 12)
        cv2.putText(annotated, text, (peak_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        if item.get("label_bbox") is not None:
            lx, ly, lw, lh = item["label_bbox"]
            cv2.rectangle(annotated, (lx, ly), (lx + lw, ly + lh), (255, 0, 255), 1)
    cv2.imwrite(save_path, annotated)
    return annotated


def main(image_path=IMAGE_PATH, show_figure=SHOW_FIGURE):
    for p, name in [(DET_MODEL_PATH, "检测模型"), (REC_MODEL_PATH, "识别模型"), (KEYS_PATH, "字符字典")]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"找不到{name}：{p}")

    detector = PPOCRDetector(DET_MODEL_PATH)
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
        label_info = read_peak_label(img_bgr, box, peak_x, peak_y, detector, recognizer)
        item = {
            "plot_index": idx,
            "roi": tuple(map(int, box)),
            "peak_pixel": (int(peak_x), round(float(peak_y), 2)),
            "label_text": None,
            "label_value": None,
            "label_bbox": None,
        }
        if label_info is not None:
            item["label_text"] = label_info.get("text")
            item["label_value"] = label_info.get("value")
            item["label_bbox"] = label_info.get("bbox")
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
        plt.title("Highest peak + label value (PP-OCR det+rec ONNX, fixed 4+4 mass format)")
        plt.tight_layout()
        plt.show()

    return results


if __name__ == "__main__":
    main()
