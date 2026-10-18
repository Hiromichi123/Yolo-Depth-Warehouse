# 部署与验收记录

日期：2026-10-18（Asia/Shanghai）。部署目录：`D:\Github\yolo-seg-global-depth`。分支：`codex/global-depth`。

## 后续需求更新：自动 ROI 与可见数计数

以下更新覆盖初始施工计划中与之冲突的推理规则，其余原始验收记录保留在后文。

- 新增根目录 `predict.py` 独立推理入口，复用 `pile-depth infer`。
- 第一步仅运行分割，按置信度/NMS/可选 `--classes` 筛选货物；密集场景 `--max-det` 默认 1000。
- 使用全部已识别货物框的外接矩形作为深度 ROI，在原始图像上裁剪并进行第二次前向；深度只使用此次 ROI 的输出。初次定位不运行深度头。
- `--roi` 变为可选搜索区域；`search_bbox` 是搜索区域，`group_bbox` 是实际货物外接框，所有坐标位于原图坐标系。
- 按用户最新要求：`total_count = floor(visible_count * average_depth)`。不再依赖 front_count；保留该参数仅作元数据。平均深度和总数排列在标注图左上角，另显示 Visible。
- 无货物时不运行深度预测，ROI、深度与总数均为 null；非空时额外保存 `.roi.png` 供核对。
- 完整自动流程由 Python 程序调度，原 ONNX 仍导出单次前向，未将动态裁剪/两次推理/最终计数写入图内。
- 最新完整测试结果：**18 passed, 1 skipped**（CUDA），`runs/pytest.xml` 已更新。新增测试覆盖货物类别筛选、外接框与原图映射、ROI 外背景不变性、空场景跳过深度、向下取整及忽略 front_count。原有模型/训练/checkpoint/ONNX 测试继续通过。
- 独立 CLI 已执行：`python predict.py --checkpoint runs/smoke/best.pt --image data/smoke/images/test/0.png --classes 0 --conf 0.001 --imgsz 64 --output runs/roi_inference/prediction.json`。合成流程输出 Visible=1、Avg Depth≈1.773327、Total=1，并生成 JPG/ROI PNG。低置信度和合成权重仅用于验证程序输出，不作为真实业务示例。
- 上传目标：`https://github.com/Hiromichi123/Yolo-Depth-Warehouse.git`。仓库只包含源代码、配置、文档、示例和依赖锁文件；虚拟环境、权重、合成数据和本机 runs 由 gitignore 排除。

## 环境与原始模型

- Python 3.11.7；独立 `.venv`，没有修改全局 Python 的模型代码。
- torch 2.5.1+cpu、torchvision 0.20.1、Ultralytics 8.3.40，完整锁文件 `requirements.lock.txt`。
- 官方 `yolo11n-seg.pt`，2,876,848 参数，COCO 80 类；随附原生模型 YAML 快照 `configs/yolo11-seg.yaml`。
- 实际 CUDA 可用性 False；`nvidia-smi` 报权限不足，因此 GPU 未验收。CPU 路径已验收。
- 原始施工计划存档 `docs/yolo_seg_global_depth_agent_plan.md`。版本环境 JSON 见 `runs/environment.json`。

## 原生结构与接口检查

YOLO11-seg 的 Backbone 在 YAML 的 backbone 字段，包含 Conv/C3k2/SPPF/C2PSA；Neck 在 head 字段，包含上采样、拼接和多尺度融合。原生 Segment 消费 P3/P4/P5。新增模块不依赖 YAML 中的固定层号，通过 Segment 实例及其 stride 最大项定位 P5，并读取实际输入通道（YOLO11n 为 256）。

| 接口 | 固定版本实际返回/用途 |
| --- | --- |
| `Segment.forward()`，training=True | `(feature_list, mask_coefficients, prototypes)` |
| `Segment.forward()`，eval | `(decoded_prediction, (feature_list, mask_coefficients, prototypes))` |
| `Segment.forward()`，export=True | `(decoded_prediction, prototypes)` |
| `BaseModel.loss()` | 调用 `SegmentationModel.init_criterion()` 的原生 `v8SegmentationLoss` |
| `v8SegmentationLoss.__call__()` | `(sum_loss * batch_size, detached_components)`，分量为 box/seg/cls/dfl |
| 原生训练/验证/预测 | `ultralytics.models.yolo.segment.train/val/predict` |
| 原生导出 | `ultralytics.engine.exporter.Exporter`；本项目提供独立 exporter |

深度输出始终 `[B,1]`，原生输出完整保留。自定义训练循环调用原生 loss 并按 batch size 归一化，新增权重为 0.2 的 SmoothL1。原生分割指标通过原生 validator 验证。没有声明原生 trainer/predictor/exporter 能自动识别自定义 checkpoint。

## 已执行验收

1. `python -m pytest -q --junitxml=runs/pytest.xml`：**12 passed, 1 skipped**。跳过项为 CUDA。包含动态 head 形状/梯度、空场景、缺失标签、公式错误、bbox 错误、polygon 错误、pile_id 泄漏、batch 对齐、冻结参数与 BN、联合梯度、checkpoint 严格重载、旧预训练模型输出逐元素保持一致、真实 bus 示例非空实例 mask、ONNX 验证。
2. 使用 `tests/conftest.py:make_dataset` 生成 train/val/test 各 2 张合成单堆图片，pile_id 相互隔离。
3. CLI 实际训练命令：`pile-depth train --data data/smoke/dataset.yaml --model yolo11n-seg.pt --depth-epochs 1 --joint-epochs 1 --batch 2 --imgsz 64 --output runs/smoke`。类别数从 80 改为 1，原生部分迁移 510/561 项。两阶段各执行一个 batch，训练后原生分割验证完成。
4. 第一阶段梯度范数：原生分割 0，深度头约 0.278；第二阶段：原生分割约 11330.829，深度头约 0.117（梯度裁剪之前）。日志如实记录，不能由此推断真实数据的最佳 loss 权重。
5. CLI `val --split test` 完成，包含深度与 box/mask mAP；见 `runs/smoke/test_metrics.json`。合成测试深度 MAE 约 0.724、总数 MAE 1.0；训练均值基线深度 MAE 0.5。box/mask mAP 为 0。这些结果仅证明工程管线可运行，不构成精度验收或优于基线的证据。
6. CLI `infer` 产出 `runs/smoke/prediction.json` 和 `.jpg`，空检测下 depth/total 为 null，front_count 不被可见数替代。
7. CLI `export --checkpoint runs/smoke/best.pt --imgsz 64 --output runs/smoke/model.onnx` 成功，ONNX checker、ONNXRuntime 的 batch 1/2 验证通过。深度最大绝对误差约 **2.38e-7**；分割 decoded 最大误差约 1.49e-4、prototype 最大误差约 5.48e-6。低阈值 NMS 每图 1 个 mask，数量、形状与二值像素完全一致。详见 `runs/smoke/model.validation.json`。

ONNX 导出出现原生 anchor 生成循环的 TracerWarning；导出空间尺寸固定，动态 batch=1/2 已实测。没有承诺空间尺寸动态、任意模型家族或真实图片上的全精度一致性。

## 交付文件

- 模型与训练：`pile_depth/model.py`、`engine.py`。
- 数据 schema/示例/校验：`examples/`、`pile_depth/data.py`。
- train/val/infer/export/check-data/env CLI：`pile_depth/cli.py`。
- OpenCV 标注与实例后处理：`pile_depth/inference.py`。
- 自定义导出：`pile_depth/exporter.py`。
- 自动化测试：`tests/`，结果 `runs/pytest.xml`。
- 依赖锁、README、模型配置、原始计划及本报告。
- 本机运行产物：官方 `yolo11n-seg.pt`；合成流程的 `runs/smoke/{best.pt,last.pt,model.onnx,history.json,...}`。模型权重、数据和 runs 已加入 gitignore。

## 尚需真实数据完成的业务验收

尚未提供真实货堆图片、实例分割标签、真实前端链数及深度标签，无法完成真实训练及业务测试。当前 smoke 权重不可用于实际货物计数。按 README 填充数据后，可直接运行两阶段训练和独立 test 验证；应按 pile_id 拆分并与固定训练均值倍率比较。GPU 验收和置信区间/不确定性估计未完成，后者按原计划属于下一版。
