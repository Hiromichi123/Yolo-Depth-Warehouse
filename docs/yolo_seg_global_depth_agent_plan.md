# YOLO-seg + Global Depth Head：Agent 施工计划

## 目标与边界
在现有 Ultralytics YOLO 实例分割模型上添加整堆平均几何深度回归分支，保留原有分割能力。单图单货堆 ROI，输出实例 mask/bbox/confidence、可见实例数、整堆平均深度；仅在已知正面独立堆叠链数 `front_count` 时计算 `total_count = round(front_count * predicted_depth)`。不得把全部可见实例数自动当作正面链数。空场景返回 `depth=None`、`total_count=None`。

## 实施前检查
1. 记录 Python、torch、ultralytics 版本和模型 YAML/权重，锁定依赖。
2. 找出当前 segmentation model 的 Backbone、Neck、Segment Head、trainer、validator、predictor、exporter 入口，检查 forward 的训练/推理/导出返回结构。
3. 在独立分支实现，优先扩展类和接口，避免修改 site-packages。增加回归头后保证旧权重能部分加载、旧模型仍能独立使用。

## 数据定义
原生 YOLO-seg 目录 images/{train,val,test}, labels/{train,val,test}；额外 depth/{train,val,test}.json，键为相对图片路径。每条样本字段：`pile_id`, `average_depth`(float>=1), `front_count`(int>=1), `total_count`(int>=front_count), `group_bbox`(xyxy原图像素)。建议可选 `camera_id`, `package_type`, `view_direction`。`average_depth` 为当前观察方向每条前后链实际包数的算术平均，`total_count = sum(chain_depths)`，且 `average_depth ≈ total_count/front_count`。数据验证脚本检查图片/标签存在、数值合法、公式一致、bbox范围、同一 pile_id 不跨拆分。可保留每链深度明细以便审计。

## 网络
共享 YOLO Backbone/Neck，在选定高语义特征图（优先 P5，按当前模型实际通道数动态确定）接 `AdaptiveAvgPool2d(1) -> Flatten -> Linear(C,128) -> ReLU -> Linear(128,1) -> 1+Softplus`。输出 `depth[B,1]`，不要取整。为回归头提供清晰的 forward 接口；不要依赖脆弱的层索引。使用货堆 ROI 图像训练和推理，尽量保留侧面与透视信息。

## 训练
1. 扩展数据集加载器，将 `depth_target`、`front_count`、`total_count` 与图像 batch 对齐。训练时需要监督深度；验证集可完整标注。若样本缺失深度标签，应显式报错或通过有效性 mask 排除，禁止默认填 0。
2. 原有 YOLO segmentation loss 原样保留，新增 `SmoothL1(pred_depth, target_depth)`；`L = L_seg + lambda_depth * L_depth`，初始 lambda_depth=0.2，记录分量和梯度尺度。第一版关闭额外 count loss。
3. 第一阶段冻结 Backbone/Neck/Seg Head，只训练 Depth Head；第二阶段解冻联合微调，Backbone 使用更低 LR。确保冻结参数和 BN 行为正确，且训练器优化器包含新增参数。
4. 禁用 Mosaic、MixUp、CopyPaste；关闭或严格限制可能改变货堆结构的随机透视、裁切。仅允许不改变标签含义的增强。谨慎处理翻转，验证当前方向定义。
5. 记录 val depth MAE/RMSE、由真实 front_count 计算的 total MAE/MAPE、分割 mAP/mask mAP，并按货物大小、深度、相机分组统计。设置对照组：固定全局平均深度倍率与深度头模型。

## 推理及可视化
输出 JSON: `visible_count`, `front_count`, `average_depth`, `total_count`, `count_source`, `group_bbox`, `instances`。若无 front_count：total_count=null，count_source='unavailable'。支持外部人工传入 front_count；将来自动前端链计数另做模块。使用 OpenCV `fillPoly` 叠加实例掩膜，绿色框和编号；蓝色货堆外框标 `total: XX` 或 `total: N/A`；左上角绿色 Hershey 字体显示 Visible、Front、Avg Depth、Total。避免文字出界。

## 模型保存与导出
checkpoint 保存模型完整 state_dict、配置、版本、深度头元数据，加载时严格校验。ONNX 导出增加深度输出，保持原生分割输出含义，验证 PyTorch/ONNXRuntime 深度数值一致、Mask 数量和形状正确。若当前 Ultralytics 导出流程不易扩展，先提供自定义 exporter，不能声称原生 exporter 已支持。

## 测试与验收
- 单元测试：深度头输出 `[B,1]`，>=1，反向传播有梯度；深度 loss 对齐 batch；空场景；缺失标签；数据集 split 泄漏；`front_count` 不存在时总数为 null。
- 集成测试：旧 YOLO-seg 权重可初始化新模型，训练 1 batch，验证 1 batch，保存/重载输出一致；原有 mask/box 仍正常；CPU/GPU 推理。
- 导出测试：ONNXRuntime 与 PyTorch 深度输出容差可配置，例如绝对误差 <1e-3（按实际算子精度调整）；分割输出一致性验证。
- 业务验收：测试集按 `pile_id` 划分；与固定倍率基线比较深度 MAE 和总数 MAE。不能保证单目完全遮挡场景准确；报告置信区间或不确定性留到下一版。

## 交付清单
1. 改造后的模型与配置代码；2. dataset schema + 校验脚本 + 示例；3. train/val/infer/export CLI；4. OpenCV 绘制模块；5. 最小单元/集成测试；6. README 命令、依赖版本、限制；7. 原版与改版兼容性说明及测试结果。

## Agent 执行顺序
先检查仓库和锁定版本 → 数据集加载及校验 → 独立深度头 forward → 冻结训练 → 联合训练 → 推理 JSON → OpenCV 标注 → checkpoint → ONNX → 全套测试。每一步运行测试并报告变更文件、命令与结果；不要仅输出设计而不修改代码。
