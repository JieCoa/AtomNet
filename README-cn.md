# Physics-informed graph neural network representation learning for crystal property prediction

# 用于晶体性质预测的物理信息增强图神经网络表示学习



<h3 align="center">
  📃 <a href="" target="_blank">Paper</a>
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
conda activate '对应环境名'
```



手动安装所有环境包

> 这是作者之前在服务器上手动配置记录，时间久远，可能遗漏部分库的安装。如果遇到部分库缺少，请自行安装补全。

```shell
conda create -n cartnet python=3.11

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



## 数据集

The datasets are automatically downloaded and processed by the code (📂`loader` → `loader.py`), except for the bulk and shear modulus that are publicly available at [Figshare](https://figshare.com/projects/Bulk_and_shear_datasets/165430). （读取文件路径记得改成自己的数据存放路径）



## 参数说明

> 我们在 main.py 设计了大量的指令参数，每个参数我们都提供了相关的描述。对于部分重要参数，我们进行详细说明，帮助 user 更好的复现我们的实验。对于未提及的参数，我们建议保留使用默认值。

- **atom_init**：用于初始化节点表示的原子描述符文件名（后缀 `.json` 会自动补全，所有原子描述符文件存放在 📂`dataset` → 📂`json`）
- **name**：当前 wandb 实验的名称，会同步到 wandb 对应 `wandb_project` 项目下生成训练记录。
  - 由于历史原因，我们对 `name` 属性字符串设置了一系列的正则表达式匹配，根据对应子串的变换，模型内部结构也会自动调整。
    1. `8five` 或 `7one` 子串出现在 `name` 属性中，表示 AtomNet Layer 中的第一个 NormLayer 使用 Layer Normalization。否则，使用 Batch Normalization。
    2. `smoothEnv02` 子串出现在 `name` 属性中，表示使用**距离权重衰减函数**（**Cubic Smooth**）。否则，使用简化的分段权重函数（**Simply**） 
    2. `newRBF, newRBF02, newRBF03, newRBF04, newRBF05` 子串出现在 `name` 属性中，表示对 Sanderson 电负性边特征使用**不同的 RBF kernel 函数**（特征维度不同，是否使用 cosine 函数进行加权等）进行特征衍生。
- **disableUpdateEdge**：<u>限制</u>或<u>停止</u>**消息传递过程**中对边特征的更新。
- **limitedUpdateEdge**：运行在前 `limitedUpdateEdge + 1` 个 AtomNet Layer 层对边特征进行更新（残差连接）。配合 `disableUpdateEdge` 参数使用，取值范围[0, 3]，对应 AtomNet Layer 层数（default: 4）。
- **usePolynomial**：默认值 `3`，我们在实验中，除了使用 RBF（径向基函数）对<u>原子间距离</u>进行**特征衍生**，还将 `特征工程` 中经典的**多项式特征衍生方法**应用于“距离”特征的扩展，虽然没有在论文中提及，但实验结果证明了该方法的有效性。（我们建议作者在所有任务中均保留使用 `--usePolynomial 3` 参数，以复现论文结果）
- **newEnvelope**：边权重计算函数。
  - 如果 user 想复现实验结果，需要为每个任务均添加 `--newEnvelope` 参数。我们分别提供了论文正文和补充文件提出的 **Cubic Smooth** 和 **Simply** 权重函数，模型选择哪个权重函数取决于上面提到的 `name` 参数。
  - 如果不添加该参数，AtomNet 将使用与 [CartNet](https://pubs.rsc.org/en/content/articlelanding/2024/dd/d4dd00352g) 相同的 cosine 函数作为权重函数。在消融实验中，我们就是通过移除 `--newEnvelope` 参数进行模型性能对比。
- **useElectronegativity**：使用基于 Sanderson 电负性作为新的边特征，
- **normalizedElectronegativity**：配合 `useElectronegativity` 参数使用，对电负性边特征进行归一化处理。
- **ig**：基于预训练模型进行解释性实验，并输出可视化结果。



### 必须修改❗

```python
parser.add_argument("--wandb_project", type=str, default="你的项目名", help="Wandb project name")
parser.add_argument("--wandb_entity", type=str, default="wandb账号名", help="Name of the wandb entity")
```





## [Weights & Biases](https://wandb.ai/site)

因为作者在模型训练过程中，使用 wandb 进行全流程记录，所以需要每个 user 在运行代码前，将 main.py 中的 `wandb_project` 和 `wandb_entity` 改成自己在 wandb 网站中创建的项目名和 wandb 账号名，并在运行环境配置 wandb 密钥。



### 密钥设置（2 种方式）

1. 写入代码中

   ```python
   import wandb 
   
   wandb.login(key=你的密钥)
   ```

2. 在**命令行设置**（适合临时使用，我用的这种方式）

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

#### Linux

> 对于 `(`  和`)` 符号，需要使用 `\` 进行转义，这是与 Windows 指令的唯一区别。

##### 1. formation energy

```shell
python main.py --seed 204 --name 8five_seed\(204\)_\(128batch_eNeg+newRBF_3_Poly_newEnv2_116d_update01\)dft_3D_formation_energy --dataset jarvis --figshare_target formation_energy_peratom --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --usePolynomial 3 --newEnvelope --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2
```



##### 2. bandgap(OPT)

```shell
python main.py --seed 448 --name smoothEnv02_8five_seed\(448\)_\(128batch_3_Poly_newEnv2_116d_update01\)dft_3D_opt_bandgap --dataset jarvis --figshare_target optb88vdw_bandgap --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --usePolynomial 3 --newEnvelope --batch 128 --disableUpdateEdge --limitedUpdateEdge 2
```



##### 3. bandgap(MBJ)

```shell
python main.py --seed 297 --name seed\(297\)_\(3_Poly_newEnv2_116d_update01\)dft_3D_mbj_bandgap --dataset jarvis --figshare_target mbj_bandgap --threads 6 --workers 3 --epochs 300 --atom_init atom_features\(116d\)_update01 --usePolynomial 3 --newEnvelope --batch 64
```



##### 4. ehull

```shell
python main.py --seed 504 --name 8five_seed\(504\)_\(128batch_eNeg+newRBF04_3_Poly_newEnv2_116d_update01\)dft_3D_ehull --dataset jarvis --figshare_target ehull --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --usePolynomial 3 --newEnvelope --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2
```



##### 5. total energy

```shell
python main.py --seed 306 --name smoothEnv02_8five_seed\(306\)_\(128batch_eNeg+newRBF_3_Poly_newEnv2_116d_update01\)dft_3D_total_energy --dataset jarvis --figshare_target optb88vdw_total_energy --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --usePolynomial 3 --newEnvelope --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2
```



#### Windows

##### 1. formation energy

```shell
python main.py --seed 204 --name 8five_seed(204)_(128batch_eNeg+newRBF_3_Poly_newEnv2_116d_update01)dft_3D_formation_energy --dataset jarvis --figshare_target formation_energy_peratom --threads 10 --workers 5 --epochs 300 --atom_init atom_features(116d)_update01 --usePolynomial 3 --newEnvelope --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2
```



##### 2. bandgap(OPT)

```shell
python main.py --seed 448 --name smoothEnv02_8five_seed(448)_(128batch_3_Poly_newEnv2_116d_update01)dft_3D_opt_bandgap --dataset jarvis --figshare_target optb88vdw_bandgap --threads 10 --workers 5 --epochs 300 --atom_init atom_features(116d)_update01 --usePolynomial 3 --newEnvelope --batch 128 --disableUpdateEdge --limitedUpdateEdge 2
```



##### 3. bandgap(MBJ)

```shell
python main.py --seed 297 --name seed(297)_(3_Poly_newEnv2_116d_update01)dft_3D_mbj_bandgap --dataset jarvis --figshare_target mbj_bandgap --threads 6 --workers 3 --epochs 300 --atom_init atom_features(116d)_update01 --usePolynomial 3 --newEnvelope --batch 64
```



##### 4. ehull

```shell
python main.py --seed 504 --name 8five_seed(504)_(128batch_eNeg+newRBF04_3_Poly_newEnv2_116d_update01)dft_3D_ehull --dataset jarvis --figshare_target ehull --threads 10 --workers 5 --epochs 300 --atom_init atom_features(116d)_update01 --usePolynomial 3 --newEnvelope --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2
```



##### 5. total energy

```shell
python main.py --seed 306 --name smoothEnv02_8five_seed(306)_(128batch_eNeg+newRBF_3_Poly_newEnv2_116d_update01)dft_3D_total_energy --dataset jarvis --figshare_target optb88vdw_total_energy --threads 10 --workers 5 --epochs 300 --atom_init atom_features(116d)_update01 --usePolynomial 3 --newEnvelope --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2
```



### Materials Project

#### Linux

##### 1. formation energy

```shell
python main.py --seed 233 --name 8five_seed\(233\)_\(128batch_eNeg_3_Poly_newEnv2_116d_update01\)megnet_formation_energy --dataset megnet --figshare_target e_form --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --usePolynomial 3 --newEnvelope --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2
```



##### 2. bandgap

```shell
python main.py --seed 335 --name smoothEnv02_8five_seed\(335\)_\(128batch_eNeg_3_Poly_newEnv2_116d_update01\)megnet_bandgap --dataset megnet  --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --usePolynomial 3 --newEnvelope --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2
```



##### 3. bulk  modulus

```shell
python main.py --seed 534 --name 8five_seed\(534\)_\(norm_eNeg_3_Poly_newEnv2_116d_update01\)megnet_bulk_modulus --dataset megnet --figshare_target 'bulk modulus' --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --usePolynomial 3 --newEnvelope --useElectronegativity --normalizedElectronegativity --batch 64 --disableUpdateEdge --limitedUpdateEdge 2
```



##### 4. shear modulus

```shell
python main.py --seed 438 --name smoothEnv02_8five_seed\(438\)_\(eNeg+newRBF02_3_Poly_newEnv2_116d_update01\)megnet_shear_modulus --dataset megnet --figshare_target 'shear modulus' --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --usePolynomial 3 --newEnvelope --useElectronegativity --batch 64 --disableUpdateEdge --limitedUpdateEdge 2
```



#### Windows

##### 1. formation energy

```shell
python main.py --seed 233 --name 8five_seed(233)_(128batch_eNeg_3_Poly_newEnv2_116d_update01)megnet_formation_energy --dataset megnet --figshare_target e_form --threads 10 --workers 5 --epochs 300 --atom_init atom_features(116d)_update01 --usePolynomial 3 --newEnvelope --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2
```



##### 2. bandgap

```shell
python main.py --seed 335 --name smoothEnv02_8five_seed(335)_(128batch_eNeg_3_Poly_newEnv2_116d_update01)megnet_bandgap --dataset megnet  --threads 10 --workers 5 --epochs 300 --atom_init atom_features(116d)_update01 --usePolynomial 3 --newEnvelope --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2
```



##### 3. bulk  modulus

```shell
python main.py --seed 534 --name 8five_seed(534)_(norm_eNeg_3_Poly_newEnv2_116d_update01)megnet_bulk_modulus --dataset megnet --figshare_target 'bulk modulus' --threads 10 --workers 5 --epochs 300 --atom_init atom_features(116d)_update01 --usePolynomial 3 --newEnvelope --useElectronegativity --normalizedElectronegativity --batch 64 --disableUpdateEdge --limitedUpdateEdge 2
```



##### 4. shear modulus

```shell
python main.py --seed 438 --name smoothEnv02_8five_seed(438)_(eNeg+newRBF02_3_Poly_newEnv2_116d_update01)megnet_shear_modulus --dataset megnet --figshare_target 'shear modulus' --threads 10 --workers 5 --epochs 300 --atom_init atom_features(116d)_update01 --usePolynomial 3 --newEnvelope --useElectronegativity --batch 64 --disableUpdateEdge --limitedUpdateEdge 2
```





## 📊可解释性结果可视化

> 可解释性实验基于预训练模型

#### 以 total energy 为例

第一步：训练模型（如果使用预训练模型，跳过）

```shell
python main.py --seed 306 --name smoothEnv02_8five_seed\(306\)_\(128batch_eNeg+newRBF_3_Poly_newEnv2_116d_update01\)dft_3D_total_energy --dataset jarvis --figshare_target optb88vdw_total_energy --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --usePolynomial 3 --newEnvelope --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2
```

第二步：进行解释性分析实验

> 就是在训练模型脚本基础上加一个 `ig` 参数

```shell
python main.py --seed 306 --name smoothEnv02_8five_seed\(306\)_\(128batch_eNeg+newRBF_3_Poly_newEnv2_116d_update01\)dft_3D_total_energy --dataset jarvis --figshare_target optb88vdw_total_energy --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --usePolynomial 3 --newEnvelope --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --ig
```

第三步：结果可视化

(1) 原子描述符中不同物理属性的重要性

> `ig_framework.py` 会通过绘制柱状图对不同原子属性重要性进行可视化。

<img src="./figure/total_energy_IG_01.png" alt="Pipeline" style="zoom:24%;" />

(2) IG 值的稳定性分析

| n1   | n2   | spearman | kendall | MRC          |
| ---- | ---- | -------- | ------- | ------------ |
| 20   | 32   | 1.0      | 1.0     | 7.364111e-05 |
| 20   | 64   | 1.0      | 1.0     | 6.908334e-05 |
| 20   | 128  | 1.0      | 1.0     | 6.897337e-05 |
| 32   | 64   | 1.0      | 1.0     | 6.235194e-06 |
| 32   | 128  | 1.0      | 1.0     | 6.304672e-06 |
| 64   | 128  | 1.0      | 1.0     | 3.072127e-07 |



## 模型效率

<img src="./figure/efficiency_comparison.png" alt="image-20260104085058662" style="zoom:46%;" />



## 引用

Please cite our paper if you find the code helpful or if you want to use the benchmark results. Thank you!



## 联系我们

如果有任何问题或建议，请联系我们 202412854021@jmu.edu.cn（第一作者）， kaihuang@jmu.edu.cn（通讯作者）。