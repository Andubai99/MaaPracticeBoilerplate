# 谱图最高峰定位与峰旁质量值 OCR 识别

## 项目目标

这个项目用于从谱图截图中自动定位蓝色曲线的最高峰，并读取峰值附近标注的质量值文本，最终输出标注结果图和结构化文本结果。

## 技术方案

核心实现位于 `find_highest_peak_and_label_ppocr_detrec_v3.py`，整体方案分为四段：

1. 图像预处理与谱图区定位
   - 使用 OpenCV 将输入图像转换到 HSV 空间。
   - 通过蓝色阈值分割提取谱图曲线区域。
   - 使用连通域分析和相邻框合并，得到谱图区 ROI。

2. 最高峰检测
   - 按列提取蓝色曲线最上缘轨迹。
   - 使用插值补全缺失列。
   - 使用 Savitzky-Golay 平滑和 `scipy.signal.find_peaks` 寻找最高峰位置。

3. 峰旁标签识别
   - 优先对峰值附近横条区域做局部字符连通域分组，直接读取完整数字串。
   - 对难例回退到 PaddleOCR 检测 + 识别双阶段链路。
   - OCR 模型采用 ONNX Runtime 推理的 PP-OCRv5 mobile det / rec 模型。
   - 通过字符归一化、数字纠错和固定 8 位数字格式规则恢复质量值。

4. 结果输出
   - 生成带 ROI、峰点、标签框和识别文本的标注图。
   - 输出 `.txt` 结果文件，记录 ROI、峰值像素、标签文本和解析后的质量值。

## 技术栈

- Python
- OpenCV
- NumPy
- SciPy
- Matplotlib
- ONNX Runtime
- PaddleOCR PP-OCRv5 mobile det / rec ONNX 模型

模型文件位于 `ocr/`，历史迭代脚本位于 `old/`。

## 当前效果

当前仓库包含两组样例结果：

- `p1_results.txt`：识别出峰旁质量值 `5484.7344`
- `p2_results.txt`：识别出峰旁质量值 `148061.99`

对应可视化标注结果见：

- `p1_annotated.png`
- `p2_annotated.png`

从现有样例看，这套方案已经具备：

- 蓝色谱图区自动定位能力
- 最高峰自动检出能力
- 峰旁质量值 OCR 识别与数值恢复能力
- 标注图与文本结果自动落盘能力

## 运行方式

安装依赖后直接运行主脚本：

```bash
python find_highest_peak_and_label_ppocr_detrec_v3.py
```

脚本当前默认读取 `p2.jpg`，并依赖以下模型文件：

- `ocr/det.onnx`
- `ocr/rec.onnx`
- `ocr/keys.txt`

## 已知说明

- 主脚本当前将 OCR 模型路径写成了本地绝对路径，迁移到其他机器时需要改成相对路径或启动参数。
- 样例结果说明方案已跑通，但泛化能力仍需更多不同截图样本继续验证。
