import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter


def merge_boxes(boxes, x_gap=10, min_y_overlap=50):
    """Merge nearby bounding boxes that belong to the same plot."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = [boxes[0][:4]]
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
    """For each x column, extract the top-most blue pixel as curve height."""
    cols = np.where(roi_mask.sum(axis=0) > 0)[0]
    if len(cols) == 0:
        return None, None

    xs, ys = [], []
    for c in cols:
        rows = np.where(roi_mask[:, c] > 0)[0]
        if len(rows) == 0:
            continue
        xs.append(c)
        ys.append(rows.min())  # smaller y = visually higher peak

    xs = np.array(xs)
    ys = np.array(ys, dtype=float)
    return xs, ys


def detect_highest_peak(image_path, show_debug=False):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Blue curve mask. This works well for screenshots similar to your example.
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Connected components: keep only the large blue components (actual curves)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blue_mask, 8)
    boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > 500:  # remove legend dots / small icons / noise
            boxes.append([x, y, x + w, y + h, area])

    plot_boxes = merge_boxes(boxes, x_gap=10, min_y_overlap=50)
    if not plot_boxes:
        raise RuntimeError("No plot area detected. Try adjusting the blue threshold or screenshot quality.")

    results = []
    annotated = img_rgb.copy()

    for idx, (x1, y1, x2, y2) in enumerate(plot_boxes, start=1):
        roi_mask = blue_mask[y1:y2, x1:x2]
        xs, ys = extract_curve_top_y(roi_mask)
        if xs is None:
            continue

        # Interpolate across gaps so we get a continuous 1D curve
        full_x = np.arange(xs.min(), xs.max() + 1)
        full_y = np.interp(full_x, xs, ys)

        # Light smoothing to suppress screenshot aliasing
        window = min(11, len(full_y) if len(full_y) % 2 == 1 else len(full_y) - 1)
        if window < 5:
            window = 5 if len(full_y) >= 5 else (len(full_y) // 2) * 2 + 1
        smooth_y = savgol_filter(full_y, window_length=window, polyorder=2)

        # High peak = smaller image y, so detect peaks on -y
        signal = -smooth_y
        peaks, props = find_peaks(signal, prominence=3, distance=15)

        if len(peaks) == 0:
            best_i = int(np.argmin(smooth_y))
        else:
            best_i = peaks[np.argmax(signal[peaks])]

        peak_x = int(full_x[best_i] + x1)
        peak_y = float(smooth_y[best_i] + y1)

        results.append({
            "plot_index": idx,
            "roi": (int(x1), int(y1), int(x2), int(y2)),
            "peak_pixel": (peak_x, round(peak_y, 2)),
        })

        # Draw result on the original image
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.circle(annotated, (peak_x, int(round(peak_y))), 8, (255, 0, 0), 2)
        cv2.putText(
            annotated,
            f"Highest peak {idx}",
            (peak_x + 10, max(20, int(round(peak_y)) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        if show_debug:
            plt.figure(figsize=(8, 3))
            plt.plot(full_x + x1, smooth_y + y1, label="curve top")
            if len(peaks) > 0:
                plt.scatter(full_x[peaks] + x1, smooth_y[peaks] + y1, s=20, label="candidate peaks")
            plt.scatter([peak_x], [peak_y], s=50, label="highest peak")
            plt.gca().invert_yaxis()
            plt.title(f"Plot {idx}")
            plt.legend()
            plt.tight_layout()
            plt.show()

    # Show final annotated result
    plt.figure(figsize=(16, 5))
    plt.imshow(annotated)
    plt.axis("off")
    plt.title("Detected highest peak in each plot")
    plt.tight_layout()
    plt.show()

    print("\nDetection results:")
    for r in results:
        print(f"Plot {r['plot_index']}: highest peak pixel = {r['peak_pixel']}, ROI = {r['roi']}")

    return results


if __name__ == "__main__":
    # 把这里改成你的截图路径
    image_path = "p1.png"

    # True 会额外显示每张子图的 1D 峰检测过程
    detect_highest_peak(image_path, show_debug=True)
