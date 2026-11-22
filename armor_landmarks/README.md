# 装甲板 ROI 分类与固定关键点原型

输入是上游已经裁剪的单块装甲板图像。输出类别、固定点 ID、原始 ROI 像素坐标、相对 ROI 中心偏移、可见性和有效标志。本项目不检测装甲板，不做 PnP，不包含云台或射击控制。

**这是可训练的研发原型，不是已验证的赛场模型。** 六张用户模板已导入。`templates/config.json` 中的点是辅助搭建程序时选择的草稿；不是用户正式提供的物理点映射。没有实拍数据、相机信息和正式点表，不能声称达到实际 PnP 精度。

## 1. 文件与结构

```text
armor/
  model.py        ResNet18 + FPN + 分类/类别专属热图/可见性
  data.py         模板校验、在线合成、实拍 JSONL 数据集
  geometry.py     变换、letterbox、热图坐标、中心偏移
  inference.py    局部亚像素解码、坐标还原、结果过滤
templates/        六张原始 PNG 和草稿配置
annotate.html     可离线打开的模板标点工具
tools/            标点页面模板与生成脚本
tests/            坐标、标签、梯度和推理接口测试
main.py           preview / train / predict / evaluate / export
```

模型输入默认 `B×3×192×192`。ResNet18 的 C2/C3/C4/C5 经 64 通道 FPN 融合成 `B×64×48×48`；分类输出 `B×7`，包含六类和无效类；热图输出 `B×6×12×48×48`，可见性输出 `B×6×12`。六类共享特征，但使用不同最终热图通道。训练按照真实类别选头，推理按照预测类别选头。

默认不自动下载预训练权重，`--pretrained` 才启用 ImageNet 权重下载。第一版未做速度压缩、量化、ONNX 或 TensorRT 验证；提供已可检验输出一致性的 TorchScript 导出路径。

## 2. 安装与快速启动

在本项目目录打开终端。建议使用与 PyTorch 支持匹配的 Python 版本。Windows / Linux 都使用相同 Python 入口。

**此电脑可直接使用已有 PyCharm/Conda 环境 `E:\anaconda3\envs\yolo\python.exe`，无需重新安装依赖。** 已检测到 CUDA 可用及 RTX 5060 Laptop GPU。PyCharm 中选择“添加本地解释器 → 已有环境”并填写该路径；项目根目录选本 README 所在的 `armor_landmarks`。

**最快查看已训练原型：直接右键运行 `demo.py`，不需要填写运行参数。** 默认处理数字 4，输出到 `runs/demo/result.json` 和 `result.png`。更换图片只需修改文件顶部的 `IMAGE` 路径。随附 `runs/prototype/best.pt` 是 10 轮合成训练权重，仍使用草稿点位；查看 `VALIDATION.md` 了解实际测试范围。

在 PyCharm 运行 `main.py` 时，工作目录设为项目根目录，参数填例如 `preview --out runs/preview` 或下面的训练参数。也可在 PowerShell 中运行：

```powershell
& 'E:\anaconda3\envs\yolo\python.exe' tests/run_tests.py
& 'E:\anaconda3\envs\yolo\python.exe' main.py preview
```

以下 venv 安装步骤仅供迁移到没有依赖的新电脑时使用。`tests/run_tests.py` 无需 pytest；`python -m pytest` 是可选替代方式。

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest -q
.venv/Scripts/python main.py preview --out runs/preview
```

本次实际验证使用上面的现有 `yolo` 环境；早期尝试建立的 `work/.venv` 未完成依赖安装，请勿选用。移动项目后请自行安装环境。GPU 训练需要与设备匹配的 PyTorch 构建；本项目不假设新安装的一定支持 CUDA。

## 3. 先确认模板点

双击 `annotate.html`。下拉框切换类别，选右侧点再点击图片移动；可增加、删除、改 ID。最多 12 点，每类至少 4 点。导出 `config.json` 后替换 `templates/config.json`。

勾选“已核对”只代表你核对了图案点位与编号，不会自动生成物理坐标。正式使用要将 `(class_id, point_id)` 与已有 PnP 物理坐标表一一对应。修改点顺序、ID 或坐标之后，应重新训练；checkpoint 内保存的配置不会随外部 JSON 自动变化。

手工编辑配置后重新生成网页：

```powershell
python tools/make_annotation_page.py
```

草稿也可用来验证管线，但需要明确指定 `--allow-draft`。不要把草稿点 ID 直接传给已有 PnP 表。

## 4. 生成可检查的合成样本

```powershell
python main.py preview --count 28 --out runs/preview
```

生成原始样本、`manifest.jsonl` 和带绿色点/红色 ID 的 `contact_sheet.jpg`。变换包括透视、旋转、尺度、ROI 偏移、背景、局部遮挡、低分辨率重采样、模糊、曝光、噪声、灰度化。类别 6 为合成无效 ROI。

点坐标跟随已知变换精确传播；被矩形遮挡或出界的点不参与热图定位损失，但参与可见性监督。不会以“猜测遮挡点”代替真实可见观测。训练与验证使用独立随机种子；每个 epoch 训练样本改变，验证样本固定。

目前合成器使用全模板平面投影，不是物理渲染器；背景较简单，未模拟真实反光、非平面变形或镜头畸变。现有外框和四个圆点也仍在模板内，模型可能依赖这些特征。新板外形不同需改进合成或加入实拍数据。大量合成变体不能替代真实测试。

## 5. 小规模打通训练

```powershell
python main.py train --allow-draft --epochs 1 --samples 56 --val-samples 14 --batch 7 --max-steps 8 --out runs/smoke
```

`--max-steps` 非零时 checkpoint 标为 `smoke_test_only`。它用于检查前向、反向、保存、评估、推理链路，不能当作实际识别模型。

正式合成基线示例（已核对标点后）：

```powershell
python main.py train --pretrained --device cuda --epochs 30 --samples 4096 --val-samples 280 --batch 32 --out runs/baseline
```

CPU 可将 `--device cuda` 改成 `--device cpu` 并降低 batch。输出 `last.pt`、按验证总损失选择的 `best.pt`、`history.json`。损失为分类交叉熵 + 空间分布热图交叉熵 + 0.25×可见性 BCE。仅对真实类别且可见的有效槽位计算热图损失。

`--init path/to/best.pt` 可在相同配置下加载权重继续优化，优化器会重新初始化；不是完整断点恢复。当前实现使用余弦学习率、梯度裁剪、CPU/GPU 训练，没有混合精度或 warmup。

## 6. 接入实拍 ROI

每行一个 JSON 的 manifest，图片路径相对 manifest 所在目录。坐标是裁剪后原始 ROI 的像素坐标，不能填模板坐标，也不能填全相机图像坐标。

```json
{"image":"images/roi_001.png","class_id":0,"points":[{"id":0,"xy":[23.2,11.4],"visible":true},{"id":1,"xy":[45.0,10.2],"visible":false}]}
```

上面仅演示字段。每条正样本必须提供该类配置中所有点 ID，不可见点也应提供 `visible:false`。无效 ROI 使用 `class_id:6, points:[]`。按视频/采集批次划分 train/val/test，避免邻帧泄漏。

```powershell
python main.py train --real-train data/train.jsonl --real-val data/val.jsonl --out runs/mixed
python main.py evaluate --checkpoint runs/mixed/best.pt --manifest data/test.jsonl --out runs/test_metrics.json
```

混合训练当前为合成与实拍数据集拼接，采样比例由样本数量决定。实拍很少时，应减少 `--samples`，避免实拍样本被大量合成样本淹没。实拍数据当前只做确定性预处理，不额外在线增强。

评估输出分类准确率、真实类别头的平均/P95 定位误差、联合类别正确的 PCK@3px、至少四个正确点比例及分类统计。坐标误差已还原到各自原始 ROI 像素单位。真实类别头指标不是端到端性能；要同时看 joint 指标。评估不按置信度过滤，以避免通过丢弃困难点美化结果；实际阈值校准需另做。

## 7. 推理与 PnP 对接

```powershell
python main.py predict --checkpoint runs/baseline/best.pt --image templates/digit_4.png --out runs/prediction.json
```

得到 JSON 和同名 PNG。类别不可信会返回空点列表；类别可信但点不足时，返回点诊断信息及 `accepted:false`。绿色为有效点，橙色为被过滤点。默认阈值是研发起点，**尚未通过实拍校准**。`accepted:true` 只表示当前简单门限通过，不代表正式 PnP 精度或草稿点编号已经合格。

```json
{
  "point_id": 0,
  "uv_roi": [35.2, 21.6],
  "offset_from_roi_center": [-14.3, -24.4],
  "visibility_score": 0.9,
  "point_score": 0.7,
  "valid": true
}
```

这是字段示例，不是模型实测输出。原点在 ROI 左上像素中心，x 向右、y 向下；中心为 `((W-1)/2,(H-1)/2)`。网络不预测中心。letterbox 使用明确的像素中心仿射矩阵，反变换由同一个矩阵求逆。

如果需要全图坐标，创建含 3×3 矩阵的 JSON，例如纯裁剪原点为 `(300,200)`：

```json
[[1,0,300],[0,1,200],[0,0,1]]
```

推理加 `--roi-to-frame data/roi_transform.json`，每个点增加 `uv_frame`。若上游还有缩放或透视矫正，必须传完整逆变换。镜头畸变与内参由后续模块保持一致；本模型不自行去畸变，也不生成物理坐标。

## 8. 导出

```powershell
python main.py export --checkpoint runs/baseline/best.pt --out runs/model.ts
```

导出 TorchScript 与配置 JSON，并比较一次示例输入上的 eager/导出输出。接口仍是三个原始张量；预处理、热图解码、阈值及逆变换要按 Python 实现对齐。此原型没有把后处理导入 TorchScript。

## 9. 下一步优先级
  
1. 用真实编号替换草稿点，并确认类别名字与工程定义。
2. 用实际相机采集独立实拍 ROI，尤其是小尺寸、模糊、斜视和误裁剪。
3. 训练基线，按原始图案像素尺寸、类别和视角分析误差。
4. 校准分类/点阈值，并和 PnP 模块做端到端误差检查。
5. 确定目标硬件后，再进行部署优化与 ONNX/TensorRT 等适配。

参考方法：SuperPoint 的已知几何变换监督、Simple Baselines 的热图定位。本项目是针对固定模板的自定义实现，不是它们的官方复现。
