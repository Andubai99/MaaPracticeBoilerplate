
import os
import re
import cv2
import numpy as np
import pytesseract
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt

# =========================
# 可改参数
# =========================
IMAGE_PATH = "p1.png"   # 改成你的截图路径
SHOW_FIGURE = True         # True: 弹出结果图；False: 只保存不显示

# Windows 如未自动找到 tesseract，可取消注释并改成你的安装路径
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def merge_boxes(boxes, x_gap=20, min_y_overlap=60):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = [list(boxes[0][:4])]
    for b in boxes[1:]:
        x1, y1, x2, y2, _ = b
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
        ys.append(rows.min())  # 图像坐标里，y 越小，峰越高

    if not xs:
        return None, None

    return np.array(xs), np.array(ys, dtype=float)


def detect_plot_boxes(blue_mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blue_mask, 8)
    boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > 500:  # 去掉图例蓝点、坐标原点小蓝图标等
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
    peaks, props = find_peaks(signal, prominence=3, distance=15)

    if len(peaks) == 0:
        best_i = int(np.argmin(smooth_y))
    else:
        # 取“视觉上最高”的候选峰
        best_i = peaks[np.argmax(signal[peaks])]

    peak_x = int(full_x[best_i] + x1)
    peak_y = float(smooth_y[best_i] + y1)
    return peak_x, peak_y


def preprocess_for_ocr(crop_bgr):
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    # 把蓝线擦成白色，尽量只保留黑色文字
    blue_mask = cv2.inRange(hsv, np.array([90, 40, 40]), np.array([145, 255, 255]))
    clean = crop_bgr.copy()
    clean[blue_mask > 0] = (255, 255, 255)

    # 放大，提升 OCR 成功率
    scale = 4
    up = cv2.resize(clean, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 黑字白底二值化
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return th, scale


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


def ocr_candidates_from_crop(crop_bgr, crop_origin_xy):
    proc, scale = preprocess_for_ocr(crop_bgr)

    data = pytesseract.image_to_data(
        proc,
        output_type=pytesseract.Output.DICT,
        config="--psm 11 -c tessedit_char_whitelist=0123456789.*"
    )

    ox, oy = crop_origin_xy
    candidates = []
    n = len(data["text"])
    for i in range(n):
        txt = data["text"][i].strip()
        conf_raw = str(data["conf"][i]).strip()
        conf = float(conf_raw) if conf_raw not in ("", "-1") else -1.0
        if not txt:
            continue

        val = parse_float_like(txt)
        if val is None:
            continue

        left = data["left"][i] / scale + ox
        top = data["top"][i] / scale + oy
        width = data["width"][i] / scale
        height = data["height"][i] / scale
        cx = left + width / 2
        cy = top + height / 2

        candidates.append({
            "text": txt,
            "value": val,
            "conf": conf,
            "bbox": (int(round(left)), int(round(top)), int(round(width)), int(round(height))),
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

        # 优先考虑峰顶上方或右上方的数字
        dx = abs(cx - peak_x)
        dy = abs(cy - peak_y)

        # 如果标签中心在峰顶下方很多，罚分
        below_penalty = 120 if cy > peak_y + 35 else 0

        # OCR 置信度越高越好；conf=-1 或 0 时不强依赖
        conf_penalty = 0 if c["conf"] < 0 else max(0, 90 - c["conf"]) * 0.5

        score = dx * 1.0 + dy * 1.2 + below_penalty + conf_penalty

        if score < best_score:
            best_score = score
            best = c

    return best


def read_peak_label(img_bgr, plot_box, peak_x, peak_y):
    x1, y1, x2, y2 = plot_box
    h, w = img_bgr.shape[:2]

    # 先做局部 OCR：围绕最高峰附近找标签
    left = max(x1, peak_x - 130)
    right = min(x2, peak_x + 180)
    top = max(0, int(peak_y) - 110)
    bottom = min(h, int(peak_y) + 60)

    local_crop = img_bgr[top:bottom, left:right]
    local_candidates = ocr_candidates_from_crop(local_crop, (left, top))
    best = choose_best_label(local_candidates, peak_x, peak_y)

    if best is not None:
        return best

    # 局部没找到，再回退到整张谱图 OCR
    plot_crop = img_bgr[y1:y2, x1:x2]
    plot_candidates = ocr_candidates_from_crop(plot_crop, (x1, y1))
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
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"读不到图片：{image_path}")

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([140, 255, 255]))

    plot_boxes = detect_plot_boxes(blue_mask)

    results = []
    for idx, box in enumerate(plot_boxes, start=1):
        peak_x, peak_y = find_highest_peak_in_plot(blue_mask, box)
        label_info = read_peak_label(img_bgr, box, peak_x, peak_y)

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

    base, ext = os.path.splitext(image_path)
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
        plt.title("Highest peak + label value")
        plt.tight_layout()
        plt.show()

    return results


if __name__ == "__main__":
    main()
