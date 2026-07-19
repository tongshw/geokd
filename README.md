# GeoKD

GeoKD 是一个用于跨视角定位（cross-view localization）的教师-学生知识蒸馏项目。当前代码基于 GeoDistill 的 XY heatmap 定位目标，去掉原始 EMA self-distillation 训练环节，改为使用固定的 GeoDistill teacher checkpoint 蒸馏一个更小的 student。

核心设定：

- Teacher：预训练 GeoDistill checkpoint，训练时冻结。
- Student：较小的 DINOv2 backbone 加通道剪枝后的 DPT localization head。
- Backbone：DINOv2 特征提取器，参数冻结。
- 任务目标：仅预测 XY 平移，不做旋转预测。
- 蒸馏信号：teacher 和 student correlation heatmap 之间的交叉熵，支持普通 CE 和 peak/hard-negative 变体。
- 支持数据集：VIGOR、KITTI。
- 分布式：支持 PyTorch DDP，推荐使用 `torchrun` 启动。

## 目录结构

```text
.
├── train_geokd.py                  # 训练、验证、测试统一入口
├── geometry.py                     # BEV 投影和几何变换工具
├── VGG.py                          # 历史网络组件，部分 legacy 模型仍会 import
├── dataset/
│   ├── VIGOR.py                    # VIGOR Dataset 和 DataLoader
│   ├── KITTI.py                    # KITTI Dataset 和 DataLoader
│   ├── config_vigor_kd.yaml        # VIGOR 实验配置
│   └── config_kitti_kd.yaml        # KITTI 实验配置
├── model/
│   ├── geokd.py                    # GeoKD 模型构建、checkpoint 加载、teacher 到 student 权重拷贝
│   ├── dino.py                     # DINOv2 wrapper，使用 torch.hub 加载 facebookresearch/dinov2
│   ├── dpt.py                      # 原始 DPT localization head
│   ├── slim_dpt.py                 # 剪枝版 DPT localization head
│   ├── loss.py                     # 蒸馏 loss
│   ├── network_vigor.py            # VIGOR localization 网络和相关 correlation 逻辑
│   ├── network_kitti_dino.py       # KITTI localization 网络和相关 correlation 逻辑
│   └── efficientnet_pytorch/       # legacy EfficientNet 代码
├── utils/
│   └── util.py                     # 随机种子、度量、可视化、坐标工具
├── teahcer_ckpt/                   # Teacher checkpoint 目录，注意目录名当前拼写就是 teahcer_ckpt
├── checkpoints/                    # Student checkpoint 输出目录
└── wandb/                          # 本地 wandb 运行记录，可按需保留或清理
```

## 环境准备


- Python 3.8 
- PyTorch、torchvision
- numpy
- opencv-python 或 opencv-python-headless
- Pillow
- PyYAML
- tqdm
- einops
- scikit-learn
- matplotlib
- pandas
- wandb，可选，仅 `logging.wandb: true` 时需要

示例安装命令：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy opencv-python pillow pyyaml tqdm einops scikit-learn matplotlib pandas wandb
```

`model/dino.py` 通过下面的方式加载 DINOv2：

```python
torch.hub.load("facebookresearch/dinov2", checkpoint_name)
```

因此第一次运行需要能访问 GitHub，或者机器上已经有可用的 torch hub 缓存。如果部署环境不能联网，需要提前准备 DINOv2 hub 缓存，或把 `model/dino.py` 改成从本地路径加载。

## 数据准备

### VIGOR

默认配置：

```yaml
data:
  dataset: vigor
  dataset_root: /data/dataset/VIGOR
```

代码期望的 VIGOR 目录大致如下：

```text
/data/dataset/VIGOR/
├── NewYork/
│   ├── panorama/
│   └── satellite/
├── Seattle/
│   ├── panorama/
│   └── satellite/
├── SanFrancisco/
│   ├── panorama/
│   └── satellite/
├── Chicago/
│   ├── panorama/
│   └── satellite/
└── splits__corrected/
    ├── NewYork/
    ├── Seattle/
    ├── SanFrancisco/
    └── Chicago/
```

`data.cross_area` 控制训练/测试城市划分：

- `false`：same-area，四个城市都参与训练和测试。
- `true`：cross-area，训练使用 `NewYork`、`Seattle`，测试使用 `SanFrancisco`、`Chicago`。

### KITTI

默认配置：

```yaml
data:
  dataset: kitti
  dataset_root: /data/dataset/KITTI
  train_file: /data/test/code/HighlyAccurate/dataLoader/train_files.txt
  test1_file: /data/test/code/HighlyAccurate/dataLoader/test1_files.txt
  test2_file: /data/test/code/HighlyAccurate/dataLoader/test2_files.txt
```

代码期望的 KITTI 目录包含：

```text
/data/dataset/KITTI/
├── satmap/
└── raw_data/
    └── <drive>/
        ├── image_02/data/
        └── oxts/data/
```

同时需要 `train_file`、`test1_file`、`test2_file` 指向对应 split 文件。`cross_area: true` 时评估使用 `test2_file`，否则使用 `test1_file`。

重要限制：当前 GeoKD 是 XY-only，KITTI 配置里的 `rotation_range` 必须为 `0`，否则 `train_geokd.py` 会直接报错。

## 权重准备

训练必须提供 teacher checkpoint：

```yaml
model:
  teacher_ckpt: /path/to/teacher.pth
```

仓库中已有的默认 VIGOR teacher 路径示例：

```text
teahcer_ckpt/translation/vigor/cross-geodistill-dino.pth
```

注意：当前目录名拼写是 `teahcer_ckpt`，配置里需要和实际路径保持一致。

Student checkpoint 用途取决于运行模式：

- 训练时 `model.student_ckpt: null`：从 teacher 切片初始化 student，前提是 `init_from_teacher: true`。
- 训练时 `model.student_ckpt: /path/to/student.pth`：从该 checkpoint 恢复训练，若 checkpoint 内包含 optimizer 和 scheduler 状态，也会一并恢复。
- 评估 student 时：必须设置 `model.student_ckpt`。
- 评估 teacher 时：使用 `model.teacher_ckpt`。

Checkpoint 默认保存到：

```text
checkpoints/<dataset>/geokd/student/
```

训练过程中会写：

- `last.pth`：每个 epoch 覆盖保存最新 student。
- `<output.name>_best.pth`：验证 mean error 更低时保存最佳 student。

## 配置文件

入口只接收一个 YAML/JSON 配置文件：

```bash
python train_geokd.py --config dataset/config_vigor_kd.yaml
```

配置会被 `train_geokd.py` 展平成一个命名空间。主要字段如下。

### data

- `dataset`：`vigor` 或 `kitti`。
- `dataset_root`：数据集根目录。
- `cross_area`：是否使用 cross-area 划分。
- `image_size`：输入图像尺寸，VIGOR satellite 会 resize 到该尺寸。
- `sat_size`、`zoom`、`bev_size`、`grid_size`：模型和数据处理相关尺寸。
- `ori_noise`：VIGOR panorama 随机方向扰动角度范围。
- `train_file`、`test1_file`、`test2_file`：KITTI split 文件。
- `shift_range_lat`、`shift_range_lon`：KITTI 平移扰动范围，单位是米。
- `rotation_range`：KITTI 旋转扰动范围。本项目必须设为 `0`。

### model

- `teacher_ckpt`：teacher checkpoint 路径，训练和 teacher 评估必填。
- `student_ckpt`：student checkpoint 路径，恢复训练或 student 评估时使用。
- `teacher_dino_model`：teacher 的 DINOv2 backbone，必须和 teacher checkpoint 匹配。
- `student_dino_model`：student 的 DINOv2 backbone。
- `student_width`：SlimDPT 通道倍率，例如 `0.5`。
- `init_from_teacher`：没有 student checkpoint 时，是否从 teacher state dict 拷贝可兼容切片初始化 student。
- `levels`：参与 loss 和验证的 DPT 输出层级，例如 `[0, 2]`。
- `channels`：legacy 网络配置字段，当前构建模型时仍保留。

支持的 DINOv2 名称包括：

- `vits14`
- `vitb14`
- `vitl14`
- `vitg14`

### distill

- `student_temp`：student heatmap softmax temperature。
- `teacher_temp`：teacher heatmap softmax temperature。
- `distill_mode`：`ce` 或 `peak_hard`。
- `uncertainty_weighting`：仅普通 `ce` 模式使用，按 teacher entropy 给样本加权。
- `uncertainty_weight_min`、`uncertainty_weight_power`：uncertainty weighting 参数。
- `peak_radius`、`peak_topk`、`peak_region_weight`：`peak_hard` 的 teacher peak 区域参数。
- `hard_negative_topk`、`hard_negative_weight`：`peak_hard` 的 student hard negative 参数。

### train / eval / optim / runtime / logging / output

- `train.train`：`true` 表示训练，`false` 表示评估。
- `train.batch_size_per_gpu`：每张 GPU 的 batch size，不是全局 batch size。
- `train.epochs`：训练 epoch 数。
- `train.seed`：随机种子。DDP 下实际 seed 会加上 rank。
- `train.clip`：梯度裁剪上限。
- `eval.target`：`teacher` 或 `student`。
- `optim.lr`、`optim.min_lr`、`optim.weight_decay`：AdamW 和 cosine scheduler 参数。
- `runtime.distributed`：是否启用 DDP。
- `runtime.gpuid`：非 DDP 时选择的 GPU id。DDP 时实际设备由 `LOCAL_RANK` 决定。
- `runtime.num_workers`：每个进程的 DataLoader worker 数。
- `logging.wandb`：是否启用 wandb。
- `logging.wandb_log_interval`：step 级 wandb 记录间隔。
- `output.save_path`：checkpoint 根目录。
- `output.name`：实验名，也用于 best checkpoint 文件名。

## 训练

先修改配置文件中的路径：

- `data.dataset_root`
- KITTI 的 `train_file`、`test1_file`、`test2_file`
- `model.teacher_ckpt`
- 如需恢复训练，设置 `model.student_ckpt`
- `train.train: true`

单卡非 DDP 可以把配置改为：

```yaml
runtime:
  distributed: false
  gpuid: [0]
```

然后运行：

```bash
python train_geokd.py --config dataset/config_vigor_kd.yaml
```

多卡 DDP 推荐使用 `torchrun`：

```bash
torchrun --nproc_per_node=4 train_geokd.py --config dataset/config_vigor_kd.yaml
```

KITTI：

```bash
torchrun --nproc_per_node=4 train_geokd.py --config dataset/config_kitti_kd.yaml
```

DDP 下全局 batch size 为：

```text
global_batch_size = train.batch_size_per_gpu * nproc_per_node
```

## 评估

把配置改为：

```yaml
train:
  train: false

eval:
  target: student
```

评估 student 时需要：

```yaml
model:
  student_ckpt: /path/to/student.pth
```

运行：

```bash
torchrun --nproc_per_node=4 train_geokd.py --config dataset/config_vigor_kd.yaml
```

评估 teacher 时：

```yaml
train:
  train: false

eval:
  target: teacher

model:
  teacher_ckpt: /path/to/teacher.pth
```

评估输出包括：

- `Test mean distance`
- `Test median distance`
- `Test FPS`

FPS 统计包括 DINO 特征提取、localization head、correlation 和 argmax，不包括 DataLoader 图像读取时间。

## wandb 记录

当 `logging.wandb: true` 时，主进程会初始化：

```text
project = geokd
name = <output.name>_<YYYYMMDD_HHMMSS>
```

训练阶段记录：

- `train/step_loss`
- `train/epoch_loss`
- `train/batch_size_per_gpu`
- `train/global_batch_size`
- `val/mean_error`
- `val/median_error`
- `val/best_mean_error`
- `val/fps`
- `optim/lr`
- `distill/*`

评估阶段记录：

- `test/<target>/mean_error`
- `test/<target>/median_error`
- `test/<target>/fps`

如果环境没有安装 wandb，请把配置里的 `logging.wandb` 设为 `false`，或者安装 `wandb`。

## 代码主流程

`train_geokd.py` 的执行顺序：

1. 读取 `--config`，把分组配置展平成 `args`。
2. 校验参数，例如 KITTI `rotation_range == 0`、`eval.target` 合法、DINO 模型名合法。
3. 初始化 DDP 或单卡设备。
4. 根据 `train.train` 分支进入训练或评估。
5. 训练时：
   - 加载并冻结 teacher。
   - 构建 student。
   - 如果有 `student_ckpt`，恢复 student、optimizer、scheduler。
   - 否则在 `init_from_teacher: true` 时从 teacher 参数切片初始化 student。
   - 分别构建 teacher/student DINOv2 wrapper，并冻结 DINO 参数。
   - 每个 batch 计算 teacher correlation heatmap 和 student correlation heatmap。
   - 用 `model/loss.py` 中的蒸馏 loss 优化 student。
   - 每个 epoch 后在验证集评估并保存 checkpoint。
6. 评估时：
   - 根据 `eval.target` 加载 teacher 或 student。
   - 运行测试集，输出 mean error、median error 和 FPS。

## 常见问题

### `torch.hub.load` 下载 DINOv2 失败

第一次运行需要访问 GitHub。可选处理方式：

- 在有网络的机器上先运行一次，让 torch hub 缓存 DINOv2。
- 设置 `TORCH_HOME` 指向已有缓存目录。
- 修改 `model/dino.py`，使用本地 clone 的 dinov2 和本地权重。

### teacher checkpoint 加载时报 shape mismatch

通常是 `model.teacher_dino_model` 和 teacher checkpoint 实际使用的 DINOv2 backbone 不一致。VIGOR 默认是 `vitb14`，KITTI 默认是 `vitl14`；如果 teacher 是别的 backbone，需要同步修改配置。

### student checkpoint 加载时报 student_width 或 backbone 不一致

`train_geokd.py` 会检查 checkpoint 中保存的 `student_width` 和 `student_dino_model`。恢复训练或评估时，配置必须和 checkpoint 对应。

### KITTI 运行时报 rotation_range 错误

当前 GeoKD 只做 XY 平移蒸馏，KITTI 的 `rotation_range` 必须设为 `0`。

### DDP 配置为 true 但直接用 python 启动

如果 `runtime.distributed: true`，必须使用 `torchrun`，例如：

```bash
torchrun --nproc_per_node=4 train_geokd.py --config dataset/config_vigor_kd.yaml
```

单进程运行请设置：

```yaml
runtime:
  distributed: false
```

### 显存不够

优先调整：

- 降低 `train.batch_size_per_gpu`。
- 使用更小的 `student_dino_model`，例如 `vits14`。
- 降低 `student_width`。
- 减少 `runtime.num_workers` 只能缓解 CPU/内存压力，通常不能显著降低 GPU 显存。

## 接手建议

- 先跑通一次 `train.train: false` 的 student 评估，确认数据、DINOv2、checkpoint 路径都正确。
- 再用很小的 `epochs` 和较小 `batch_size_per_gpu` 跑训练 smoke test。
- 如果要换 teacher，必须确认 `teacher_dino_model` 和 checkpoint 完全匹配。
- 如果要比较实验，固定 `output.name`、`distill_mode`、`student_width`、`student_dino_model`，并记录全局 batch size。
- 建议后续补一个正式 `requirements.txt` 或 `environment.yml`，避免环境只能从 wandb 记录和 import 反推。
