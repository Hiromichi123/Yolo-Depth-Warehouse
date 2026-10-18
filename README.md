# YOLO-seg + Global Depth Head

在 Ultralytics YOLO11-seg 上添加整堆平均几何深度分支，保留原生实例分割和损失。开发分支：`codex/global-depth`。项目未修改 site-packages。

## 本机运行

PowerShell：

```powershell
cd D:\Github\yolo-seg-global-depth
.\.venv\Scripts\Activate.ps1
pile-depth env
python -m pytest -q
```

也可始终使用 `.\.venv\Scripts\python.exe -m pile_depth.cli`，不依赖激活脚本。

固定主版本：Python 3.11、torch 2.5.1、torchvision 0.20.1、ultralytics 8.3.40、numpy 1.26.4、OpenCV 4.10.0.84、ONNX 1.17.0、ONNXRuntime 1.20.1。完整已安装依赖见 `requirements.lock.txt`。当前虚拟环境使用 CPU PyTorch；GPU 测试仅在 CUDA 可用时执行。换 GPU 环境需安装匹配硬件的 PyTorch/CUDA 构建并重新验收，不能仅指定 `--device cuda:0`。

重建环境：

```powershell
uv venv --python 3.11 .venv
uv pip sync --python .venv\Scripts\python.exe requirements.lock.txt
uv pip install --python .venv\Scripts\python.exe --no-deps -e .
```

## 数据与标签

每张训练图已裁成**单堆 ROI**，尽量保留侧面和透视信息，裁剪范围应与推理使用的货物外接矩形保持一致。训练数据加载器不自动裁切 `group_bbox`，防止分割多边形与图像错位。标注坐标均以 ROI 图片为基准；如果原始图片还有其他货堆，必须先裁切并同步转换多边形和 bbox。推理默认自动定位货物并裁剪外接矩形；`--roi x1 y1 x2 y2` 仅限制第一步的搜索区域，最终深度 ROI 仍由检测到的货物框决定。结果坐标映射回原图。

```text
data/piles/
  dataset.yaml
  images/{train,val,test}/*.png
  labels/{train,val,test}/*.txt
  depth/{train,val,test}.json
```

配置示例 `examples/dataset.yaml`，深度标注示例 `examples/train.json`，结构定义 `examples/depth.schema.json`。JSON 键使用相对数据根目录的 POSIX 路径，例如 `images/train/pile_001.png`。每个拆分中的图片必须具有深度标签和非空分割标签。空场景仅支持推理，不能用零深度填充训练样本。

`average_depth` 是当前观察方向各独立前后链的实际包数的算术平均。必须与 `total_count/front_count` 一致（相对/绝对容差 1e-3）；`front_count` 是正面独立链数，不能使用全部可见实例数代替。按 `pile_id` 隔离 train/val/test。

```powershell
pile-depth check-data --data .\data\piles\dataset.yaml
```

校验内容：图片、polygon 标签、类别范围、深度与计数合法性、计数公式、bbox 范围、拆分泄漏、可选 chain_depths 一致性。JSON schema 本身不执行跨字段公式检查，CLI 校验器执行。

## 训练与验证

```powershell
pile-depth train --data .\data\piles\dataset.yaml --model .\yolo11n-seg.pt --depth-epochs 10 --joint-epochs 40 --batch 4 --imgsz 640 --device cpu --output runs\pile
pile-depth val --data .\data\piles\dataset.yaml --checkpoint runs\pile\best.pt --split test --output runs\test_metrics.json
```

默认 `--model yolo11n-seg.pt` 可以触发 Ultralytics 下载 COCO 预训练权重；离线应先准备该文件。`--model yolo11n-seg.yaml` 表示随机初始化，只适合管线测试。类别数变化时由原生 `SegmentationModel.load()` 部分迁移匹配权重；深度头随机初始化。旧 YOLO 权重可继续直接用于原版 `YOLO(...).predict()`。

原生损失保持不变，按 batch size 归一化后加 `0.2 * SmoothL1`。深度为连续 `[B,1]`，`1 + Softplus`，不取整。第一阶段冻结全部原生分割参数和 BN running statistics，仅优化深度头；第二阶段联合训练，原生网络学习率为深度头的 0.1 倍，固定 DFL 积分参数保持冻结。不添加 count loss。

目前关闭全部随机图像增强，仅保留原生 letterbox/格式化，包含禁止 Mosaic、MixUp、CopyPaste、翻转、透视和随机裁切。优化器显式包含深度头。`history.json` 记录深度/分割损失和梯度范数、每 epoch 深度 MAE/RMSE、真实 front_count 下 total MAE/MAPE、相机/类型/方向/深度区间/ROI面积分组指标。基线使用**训练集**平均深度，避免验证数据泄漏。

训练结束针对深度 MAE 最优 checkpoint 运行原生分割验证，保存 `segmentation_metrics.json`（box 和 mask mAP）。验证 CLI 同时给出深度与分割指标。分组中的 small/large 是 ROI 像素面积阈值 640²，不代表实际物理货物尺寸。MAPE 为比例值。真实业务效果需要真实标注数据，本项目的合成数据测试仅验证工程链路。

## 推理与标注

推理程序为根目录 `predict.py`，也可以用等价的 `pile-depth infer` 命令。完整流程：

1. 原图或手动搜索区域运行实例分割，经过置信度筛选与 NMS，保留货物类别的实例。
2. 对全部保留的货物框计算外接矩形：`[min(x1), min(y1), max(x2), max(y2)]`，左上向下取整、右下向上取整，限制在图像范围内。
3. 从原始图像裁剪这个矩形，重新 letterbox 后输入模型；仅使用此次 ROI 前向得到的平均深度。
4. 用第一步的识别数量计算 `total_count = floor(visible_count * average_depth)`，结果与平均深度同时显示在推理图左上角。

第一步不运行深度头。第二步深度计算保留 ROI 内货物间隙和背景，不对 mask 内像素分别求深度。实例 mask/bbox 和 Visible 来自第一步，不用第二次前向结果覆盖，以保证 ROI 来源与计数一致。默认一次处理一堆密集货物，多个相隔很远的货堆会合并为一个外接框，必要时通过 `--roi` 指定单堆搜索区域。

```powershell
python predict.py --checkpoint runs\pile\best.pt --image pile.jpg --classes 0 --output runs\prediction.json
pile-depth infer --checkpoint runs\pile\best.pt --image pile.jpg --classes 0 --conf 0.25 --iou 0.7 --max-det 1000 --output runs\prediction.json
pile-depth infer --checkpoint runs\pile\best.pt --image frame.jpg --roi 100 80 900 700 --output runs\roi.json
```

`--classes` 指定货物类别 ID（可以传多个）；货物专用模型可省略，省略时保留模型的所有类别。若使用多类别模型，应选择货物对应的类别，避免无关目标扩大 ROI。`--max-det` 默认 1000，可按密集程度调整。

输出 JSON 包含 `visible_count`, `front_count`, `average_depth`, `total_count`, `count_source`, `group_bbox`, `instances`，实例包括 bbox/confidence/class_id/mask_polygons。`group_bbox` 是实际深度裁剪框，`search_bbox` 是第一步搜索区域，`depth_roi_source=detected_cargo_union`。同名 JPG 是绿色 mask、框和编号、蓝色货堆框，左上角按 Avg Depth、Total、Visible 排列。非空场景另保存同名 `.roi.png`，便于核对实际深度输入。未识别到货物时跳过深度前向，`group_bbox`、深度和总数均为 null，图中文字为 N/A。

按最新需求，推理总数使用**全部识别货物数乘连续平均深度，并向下取整**。例如 Visible=3、Avg Depth=2.6，Total=7，`count_source=visible_count_x_average_depth_floor`。兼容保留 `--front-count` 元数据，但它不再影响推理总数。这项计数规则覆盖桌面初始计划的 front_count 规则；训练深度标签定义和验证中基于真实 front_count 的对照指标保持原定义。推理未做置信区间/不确定性估计。

## 保存与 ONNX

```powershell
pile-depth export --checkpoint runs\pile\best.pt --output runs\pile.onnx --imgsz 640 --atol 0.001
```

checkpoint 保存完整 state_dict、模型 YAML、类别名、版本、深度头元数据、训练记录、训练集深度基线；加载使用 weights_only 并严格核对 state_dict、格式、Ultralytics 版本和头元数据。这是自定义 checkpoint，应使用本项目加载器，不能直接当作原生 YOLO `.pt`。

使用**自定义 exporter**，未扩展原生 Ultralytics exporter。输出为原生 segmentation `[B,4+nc+32,N]`、prototypes `[B,32,H/4,W/4]`、average_depth `[B,1]`；前两个输出含义保持原样，需要 NMS 和 mask 后处理。固定导出空间尺寸，动态 batch。CLI 导出后强制用 ONNXRuntime 校验 batch=1/2 的所有输出、NMS 结果和二值 mask，写入 `.validation.json`。默认 atol=1e-3、rtol=1e-4；测试输入为随机数，实际部署前还应使用真实 ROI 验证。

ONNX 图是单次前向，不包含自动外接框裁剪、两阶段调度或最终计数。当前 `predict.py` 使用 PyTorch 执行完整两阶段流程；ONNX 部署需在宿主程序中实现相同的检测、裁剪和二次前向。

## 实现入口与边界

- `pile_depth/model.py`：Segment 输入 pre-hook、动态通道深度头、损失、冻结、checkpoint。
- `pile_depth/data.py`：原生 YOLODataset 扩展、batch 对齐、严格数据校验。
- `pile_depth/engine.py`：两阶段训练、自定义深度验证和原生分割验证。
- `pile_depth/inference.py`：原生 NMS/prototypes 后处理、ROI 映射、JSON 与 OpenCV 绘制。
- `pile_depth/exporter.py`：独立 ONNX 导出和数值验收。
- `tests/`：标签错误、空场景、batch 对齐、冻结/BN、联合梯度、重载、CPU/CUDA、ONNX 测试。

对照的固定版原生入口：`ultralytics.nn.tasks.SegmentationModel`、`ultralytics.nn.modules.head.Segment`、`ultralytics.models.yolo.segment.{train, val, predict}`、`ultralytics.engine.exporter.Exporter`。实际源码接口与检查记录见 `IMPLEMENTATION_REPORT.md`。官方说明：[Segmentation](https://docs.ultralytics.com/tasks/segment/)、[Segment 源码文档](https://docs.ultralytics.com/reference/nn/modules/head/)。项目继承 Ultralytics 使用条件；分发或商业部署请按原依赖许可证处理。
