# Physics-informed graph neural network representation learning for crystal property prediction

# 用于晶体性质预测的物理信息增强图神经网络表示学习

[![License: MIT](http://typora-image-management.oss-cn-hangzhou.aliyuncs.com/img/License-MIT-yellow.svg)](https://github.com/JieCoa/AtomNet/blob/main/LICENSE)

<h3 align="center">
  📃 <a href="https://www.nature.com/articles/s41524-026-02131-9" target="_blank">Paper</a>
</h3>



## AtomNet 架构图

![Pipeline](./figure/AtomNet_structure.png)




## 环境配置

> 国内用户建议在安装环境前，提前将 conda 源换成阿里、清华、中科大镜像源，具体 url 请自行搜索。

使用仓库中的环境配置文件 `environment.yml` 自动安装

```shell
# Create a Conda environment
conda env create -f environment.yml

# Activate the environment
conda activate 'atomnet'
```



手动安装所有环境包

> 这是作者之前在服务器上手动配置记录，时间久远，可能遗漏部分库的安装。如果遇到部分库缺少，请自行安装补全。

```shell
conda create -n atomnet python=3.11

conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c nvidia
# 如果遇到NVIDIA 官方 Conda channel 的 bug（包文件损坏或元数据校验错误），全球用户都会遇到。使用下面命令↓，pytorch-cuda=12.1 默认依赖 cuda-version >=12,<13，会去取最新（即 12.9）；我们手动固定 cuda-version=12.1 → 规避损坏的 12.9-3 包。
# conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 cuda-version=12.1 -c pytorch -c nvidia -c conda-forge

# 安装torch_geometric的依赖
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html

# 安装torch_geometric
pip install torch-geometric

# 安装 jarvis 包
pip install jarvis-tools

# 根据代码补全剩余的包
pip install roma
pip install pandas
pip install pytorch_lightning
pip install yacs
pip install wandb
pip install mendeleev==0.14.0
pip install captum


# 放最后：不兼容 numpy2.X 以上版本，要换回 1.23.X 以上，但 1.26.0 以下版本
pip install numpy==1.26.0
```



## ⭐数据集

大部分数据集会自动联网下载，并通过代码 (📂`loader` → `loader.py`)进行数据预处理，在 [Figshare](https://figshare.com/projects/Bulk_and_shear_datasets/165430) 上公开提供的 **bulk modulus** 和 **shear modulus** 数据集，需要手动下载并存放到 `dataset/megnet/` 路径下。（读取文件路径记得改成自己的数据存放路径）

我们也提供预处理的数据集，以便您可以直接进行模型训练或推理。（实际上，数据预处理过程仅需几分钟。）

❗ 将**预处理的数据集**放置在 `dataset/jarvis/preprocessed/` 目录中。

- 🔎 [Jarvis DFT 3D 2021](https://doi.org/10.5281/zenodo.18993843)
- 🔎 [Jarvis megnet (Materials Project 2018)](https://zenodo.org/records/20027439)



## ⭐Pre-trained models

对于 Jarvis 数据集中的五项任务，我们提供了相应的 [预训练模型](https://zenodo.org/records/19045099) 用于推理实验。

❗ 解压压缩包，并将 📂`results` 文件夹 放到项目文件的根目录。

- 简而言之，就是将**预训练模型**放到 `results/` 目录中。



## 参数说明

> 我们在 main.py 设计了大量的指令参数，每个参数我们都提供了相关的描述。对于部分重要参数，我们进行详细说明，帮助 user 更好的复现我们的实验。对于未提及的参数，我们建议保留使用默认值。

- **atom_init**：用于初始化节点表示的原子描述符文件名（后缀 `.json` 会自动补全，所有原子描述符文件存放在 📂`dataset` → 📂`json`）
- **name**：当前 wandb 实验的名称，会同步到 wandb 对应 `wandb_project` 项目下生成训练记录。
- **electronegativity_type**：对 Sanderson 电负性边特征使用**不同的 RBF kernel 函数**（特征维度不同，是否使用 cosine 函数进行加权等）
- **envelope_type**：我们分别提供了论文正文和补充文件提出的 **Cubic Smooth** 和 **Simply** 权重函数，即 `cubic` 和 `simply`。
- **disableUpdateEdge**：<u>限制</u>或<u>停止</u>**消息传递过程**中对边特征的更新。
- **limitedUpdateEdge**：运行在前 `limitedUpdateEdge + 1` 个 AtomNet Layer 层对边特征进行更新（残差连接）。配合 `disableUpdateEdge` 参数使用，取值范围[0, 3]，对应 AtomNet Layer 层数（default: 4）。
  - 在 Jarvis DFT 3D 2021 数据集的 BandGap(MBJ) 任务中，我们发现当设置 `limitedUpdateEdge==3` 时（如果 AtomNet_layer 为 4 层），模型的 MAE 指标更优。或者直接在模型训练脚本中删除 `--disableUpdateEdge` 和 `--limitedUpdateEdge`。
- **usePolynomial**：默认值 `3`，我们在实验中，除了使用 RBF（径向基函数）对<u>原子间距离</u>进行**特征衍生**，还将 `特征工程` 中经典的**多项式特征衍生方法**应用于“距离”特征的扩展，虽然没有在论文中提及，但实验结果证明了该方法的有效性。（我们建议作者在所有任务中均保留使用 `--usePolynomial 3` 参数，以复现论文结果）
- **useElectronegativity**：使用基于 Sanderson 电负性作为新的边特征，
- **normalizedElectronegativity**：配合 `useElectronegativity` 参数使用，对电负性边特征进行归一化处理。
  - 当使用该参数时，会自动忽视 `electronegativity_type` 的参数值，即不对电负性边特征进行维度扩展。
- **inference**：基于预训练模型使用推理学习。
  - 由于模型的初始化依赖于预训练模型的具体架构细节。所以，我们建议 user 使用我们提供的示例代码和预训练模型进行尝试。如果希望用自己的训练模型，需要先用一套模型训练脚本生成预训练模型，再在这套训练脚本中加上 `--inference` 即可实现推理学习。
- **ig**：基于预训练模型进行解释性实验，并输出可视化结果。
  - 与 `inference` 流程一样，在得到预训练模型的那套脚本中加上 `--ig` 参数，或使用我们提供的示例代码和预训练模型，即可实现针对原子描述符特征的可解释性分析结果。
- **max_neighbours**：经过实验验证，虽然限制中心原子的邻居数量能够一定程度上减少模型的参数量和训练时间，但在最终的 MAE 指标上，仅依赖截止半径的预测结果优于使用截断半径与最大邻居数双重限制的模型性能。



### 必须修改❗

> 我们强烈推荐使用 [wandb](https://wandb.ai/site) 来监控模型训练。这是一个非常出色的可视化网站。

```python
parser.add_argument("--wandb_project", type=str, default="你的项目名", help="Wandb project name")
parser.add_argument("--wandb_entity", type=str, default="wandb账号名", help="Name of the wandb entity")
```





## [Weights & Biases](https://wandb.ai/site)

因为我们在模型训练过程中，使用 wandb 进行全流程记录，所以需要每个 user 在运行代码前，将 main.py 中的 `wandb_project` 和 `wandb_entity` 改成自己在 wandb 网站中创建的项目名和 wandb 账号名，并在运行环境配置 wandb 密钥。



### 密钥设置（2 种方式）

1. 写入代码中，e.g. `train.py`

   ```python
   import wandb 
   
   wandb.login(key=你的密钥)
   ```

2. 在**命令行设置**（适合临时使用）

   ```shell
   export WANDB_API_KEY=你的密钥
   ```



### 训练记录上传 wandb （仅适用于 `offline` 模式，模型训练结束后再执行）

因为服务器的访问限制，我们在模型训练过程中使用 `offline` 模式，需要在模型训练完成后，手动进行记录上传。

```python
# wandb 离线模式(如果无法连接 wandb 服务器，请使用该模式)
run = wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=cfg.name, config=cfg, mode="offline")

# wandb 在线模式(remove 'mode="offline"')
run = wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=cfg.name, config=cfg)
```

上传指令：

1. 打开终端，启动虚拟环境（需要安装 wandb 包）
2. 进入 `wandb` 日志存储路径（项目中的 📂`wandb` 文件夹）
3. 执行 `wandb sync "实验记录文件夹名"` 进行记录上传。（上传前确保能够正常访问 `wandb` 网站，否则会提示超时）
4. 如果上传成功，`"实验记录文件夹名"` 的目录下会生成一个 `xxx.synced` 文件。



## 🚀训练

为了重现论文中的实验，我们为 user 提供每个任务的最优模型训练配置代码（论文结果由 10 个不同 seed 参数训练得到的测试集 MAE 值，删除一个最大值和一个最小值，再对剩余 8 个结果计算平均值得到）。



### Jarvis DFT 3D 2021

我们提供了 2 种脚本执行方式：

1. 在命令行终端执行脚本（示例脚本如下所示）；
2. Linux 系统执行 `/scripts/linux_train_atomnet.py`。



#### Linux

> 对于 `(`  和`)` 符号，需要使用 `\` 进行转义，这是与 Windows 指令的唯一区别。

##### 1. formation energy

```shell
python main.py --seed 206 --name dft_3D_formation_energy --dataset jarvis --figshare_target formation_energy_peratom --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type simply --electronegativity_type newRBF
```

##### 2. bandgap(OPT)

```shell
python main.py --seed 448 --name dft_3D_opt_bandgap --dataset jarvis --figshare_target optb88vdw_bandgap --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type cubic
```

##### 3. bandgap(MBJ)

```shell
python main.py --seed 296 --name dft_3D_mbj_bandgap --dataset jarvis --figshare_target mbj_bandgap --epochs 300 --atom_init atom_features\(116d\)_update01 --envelope_type simply
```

##### 4. ehull

```shell
python main.py --seed 510 --name dft_3D_ehull --dataset jarvis --figshare_target ehull --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type cubic --electronegativity_type newRBF04
```

##### 5. total energy

```shell
python main.py --seed 306 --name dft_3D_total_energy --dataset jarvis --figshare_target optb88vdw_total_energy --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type cubic --electronegativity_type newRBF
```



#### Windows

##### 1. formation energy

```shell
python main.py --seed 206 --name dft_3D_formation_energy --dataset jarvis --figshare_target formation_energy_peratom --threads 10 --workers 5 --epochs 300 --atom_init atom_features(116d)_update01 --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type simply --electronegativity_type newRBF
```

##### 2. bandgap(OPT)

```shell
python main.py --seed 448 --name dft_3D_opt_bandgap --dataset jarvis --figshare_target optb88vdw_bandgap --threads 10 --workers 5 --epochs 300 --atom_init atom_features(116d)_update01 --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type cubic
```

##### 3. bandgap(MBJ)

```shell
python main.py --seed 296 --name dft_3D_mbj_bandgap --dataset jarvis --figshare_target mbj_bandgap --epochs 300 --atom_init atom_features(116d)_update01 --envelope_type simply
```

##### 4. ehull

```shell
python main.py --seed 510 --name dft_3D_ehull --dataset jarvis --figshare_target ehull --threads 10 --workers 5 --epochs 300 --atom_init atom_features(116d)_update01 --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type cubic --electronegativity_type newRBF04
```

##### 5. total energy

```shell
python main.py --seed 306 --name dft_3D_total_energy --dataset jarvis --figshare_target optb88vdw_total_energy --threads 10 --workers 5 --epochs 300 --atom_init atom_features(116d)_update01 --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type cubic --electronegativity_type newRBF
```



### Materials Project

> 这里我们统一提供适用于 linux 的运行脚本。如果想改成 windows，直接删除 `--atom_init` 部分的 `\`。

##### 1. e_form

```shell
python main.py --seed 235 --name megnet_formation_energy --dataset megnet --figshare_target e_form --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type simply
```

##### 2. bandgap

```shell
python main.py --seed 331 --name megnet_bandgap --dataset megnet --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type cubic
```

##### 3. bulk  modulus

```shell
python main.py --seed 536 --name megnet_bulk_modulus --dataset megnet --figshare_target 'bulk modulus' --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --normalizedElectronegativity --batch 64 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type simply
```

##### 4. shear modulus

```shell
python main.py --seed 440 --name megnet_shear_modulus --dataset megnet --figshare_target 'shear modulus' --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 64 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type cubic --electronegativity_type newRBF02
```



## 推理

### 数据

> 理想情况下，用于推理的晶体结构应区别于训练数据集，且推理数据集的数据结构应与我们提供的 `inference_data.json` 保持一致。
>
> - 与模型训练数据集一样，我们统一使用 `from jarvis.core.atoms import Atoms` 包处理推理数据，因此要求满足下面展示的 json 数据结构。
> - "props" 可以为空，不会在数据处理过程中使用。

#### inference_data.json

```json
// "jid": "JVASP-23213"
[
    {"atoms": {
        "lattice_mat": [
            [
                1.6374061181864787,
                4.6320620294223085,
                2.836071848582606
            ],
            [
                -0.0003656271176311,
                -0.0002587898763079,
                5.6727763751083975
            ],
            [
                4.9133178857659425,
                8.84592966e-08,
                -2.836705415419651
            ]
        ],
        "coords": [
            [
                3.2751799999999998,
                2.3159,
                2.8360700000000003
            ],
            [
                0.81852,
                2.3159,
                4.254425
            ],
            [
                2.45666,
                0.0,
                -1.418355
            ],
            [
                3.275365,
                2.31603,
                -0.000320000000000098
            ],
            [
                0.8187019541225831,
                0.5789094103158883,
                1.4180396205680705
            ],
            [
                5.731658045877419,
                4.052890589684114,
                4.254100379431928
            ],
            [
                2.564110246226763,
                3.4359435249916004,
                1.4178436178370966
            ],
            [
                2.456217182400851,
                3.3596525149154344,
                4.254296382471964
            ],
            [
                1.6914174084098796,
                1.1960121133920303,
                2.929622058350195
            ],
            [
                4.094142817599149,
                1.2721474850845667,
                1.417843617528035
            ],
            [
                1.691196226115134,
                1.1958564755409065,
                5.579347730486713
            ],
            [
                4.858942591590121,
                3.435787886607971,
                2.7425179416498056
            ],
            [
                3.986249753773237,
                1.195856475008399,
                4.254296382162902
            ],
            [
                4.859163773884865,
                3.4359435244590935,
                0.09279226951328705
            ]
        ],
        "elements": [
            "Fe",
            "Fe",
            "Fe",
            "Fe",
            "Fe",
            "Fe",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O"
        ],
        "cartesian": true,
        "props": [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
        ]
    }}
]
```



### 示例脚本

> 这里我们统一提供适用于 linux 的运行脚本。如果想改成 windows，直接删除 `--atom_init` 部分的 `\`。

#### Jarvis DFT 3D 2021

##### 1. formation energy

```shell
python main.py --seed 206 --name dft_3D_formation_energy --dataset jarvis --figshare_target formation_energy_peratom --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type simply --electronegativity_type newRBF --inference
```

##### 2. bandgap(OPT)

```shell
python main.py --seed 448 --name dft_3D_opt_bandgap --dataset jarvis --figshare_target optb88vdw_bandgap --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type cubic --inference
```

##### 3. bandgap(MBJ)

```shell
python main.py --seed 296 --name dft_3D_mbj_bandgap --dataset jarvis --figshare_target mbj_bandgap --epochs 300 --atom_init atom_features\(116d\)_update01 --envelope_type simply --inference
```

##### 4. ehull

```shell
python main.py --seed 510 --name dft_3D_ehull --dataset jarvis --figshare_target ehull --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type cubic --electronegativity_type newRBF04 --inference
```

##### 5. total energy

```shell
python main.py --seed 306 --name dft_3D_total_energy --dataset jarvis --figshare_target optb88vdw_total_energy --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type cubic --electronegativity_type newRBF --inference
```



#### Materials Project

##### 1. e_form

```shell
python main.py --seed 235 --name megnet_formation_energy --dataset megnet --figshare_target e_form --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type simply --max_neighbours -1 --inference
```

##### 2. bandgap

```shell
python main.py --seed 331 --name megnet_bandgap --dataset megnet --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type cubic --max_neighbours -1 --inference
```

##### 3. bulk  modulus

```shell
python main.py --seed 536 --name megnet_bulk_modulus --dataset megnet --figshare_target 'bulk modulus' --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --normalizedElectronegativity --batch 64 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type simply --inference
```

##### 4. shear modulus

```shell
python main.py --seed 440 --name megnet_shear_modulus --dataset megnet --figshare_target 'shear modulus' --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 64 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type cubic --electronegativity_type newRBF02 --inference
```



## 📊可解释性结果可视化

> 可解释性实验基于预训练模型。这里我们统一提供适用于 linux 的运行脚本。如果想改成 windows，直接删除 `--atom_init` 部分的 `\`。

#### 以 total energy 为例

第一步：训练模型（如果使用预训练模型，跳过）

```shell
python main.py --seed 306 --name dft_3D_total_energy --dataset jarvis --figshare_target optb88vdw_total_energy --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type cubic --electronegativity_type newRBF
```

第二步：进行解释性分析实验

> 就是在训练模型脚本基础上加一个 `ig` 参数

```shell
python main.py --seed 306 --name dft_3D_total_energy --dataset jarvis --figshare_target optb88vdw_total_energy --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type cubic --electronegativity_type newRBF --ig
```

第三步：结果可视化

(1) 原子描述符中不同物理属性的重要性

> `ig_framework.py` 会通过绘制柱状图对不同原子属性重要性进行可视化。

<img src="./figure/total_energy_IG_01.png" alt="Pipeline" style="zoom:26%;" />

(2) IG 值的稳定性分析

| n1   | n2   | spearman | kendall | MRC      |
| ---- | ---- | -------- | ------- | -------- |
| 20   | 32   | 1.0      | 1.0     | 0.000186 |
| 20   | 64   | 1.0      | 1.0     | 0.000196 |
| 20   | 128  | 1.0      | 1.0     | 0.000196 |
| 32   | 64   | 1.0      | 1.0     | 0.000026 |
| 32   | 128  | 1.0      | 1.0     | 0.000026 |
| 64   | 128  | 1.0      | 1.0     | 0.000002 |



## 模型效率

<img src="./figure/efficiency_comparison.png" alt="image-20260104085058662" style="zoom:46%;" />



## 引用

如果您认为我们的代码有帮助，或者想使用这些基准测试结果，请引用[我们的论文](https://www.nature.com/articles/s41524-026-02131-9)。非常感谢！

```bibtex
# BibTex
@article{cao2026physics,
  title={Physics-informed graph neural network representation learning for crystal property prediction},
  author={Cao, Jie and Huang, Kai and Mao, Jian and Zhong, Mengya and Chen, Chen and Li, Kexun and Liu, Taikang},
  journal={npj Computational Materials},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
```



## 联系我们

如果有任何问题或建议，请联系我们 202412854021@jmu.edu.cn（第一作者）， kaihuang@jmu.edu.cn（通讯作者）。