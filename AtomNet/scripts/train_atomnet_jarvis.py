import os

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 切换到上一级目录
os.chdir("..")

jarvis = [
    # ----- jarvis -----
    # 🚀形成能
    # "python main.py --seed 212 --name seed\(212\)_\(atom_number\)jarvis_dft_3D_formation_energy_peratom --dataset jarvis "
    # "--figshare_target formation_energy_peratom --threads 6 --workers 3 --epochs 300",

    # "python main.py --seed 203 --name 7one_seed\(203\)_\(116d_update01\)dft_3D_formation_energy --dataset jarvis"
    # " --figshare_target formation_energy_peratom --threads 6 --workers 3 --epochs 300 --atom_init atom_features\(116d\)_update01",

    "python main.py --seed 210 --name smoothEnv02_8five_seed(210)_(128batch_3_Poly_newEnv2_116d_update01)dft_3D_formation_energy --dataset jarvis "
    "--figshare_target formation_energy_peratom --threads 10 --workers 5 --epochs 300 --atom_init atom_features(116d)_update01"
    " --usePolynomial 3 --newEnvelope --batch 128 --disableUpdateEdge --limitedUpdateEdge 2",

    # ---- ⭐ useElectronegativity ⭐ ----
    # "python main.py --seed 204 --name seed\(204\)_\(eNeg+newRBF_116d_update01\)dft_3D_formation_energy --dataset jarvis"
    # " --figshare_target formation_energy_peratom --threads 6 --workers 3 --epochs 300 --atom_init atom_features\(116d\)_update01"
    # " --useElectronegativity",

    # "python main.py --seed 210 --name smoothEnv02_8five_seed\(210\)_\(128batch_eNeg_3_Poly_newEnv2_116d_update01\)dft_3D_formation_energy --dataset jarvis "
    # "--figshare_target formation_energy_peratom --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01"
    # " --usePolynomial 3 --newEnvelope --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2",

    # "python main.py --seed 212 --name 8five_seed\(212\)_\(128batch_norm_eNeg_3_Poly_newEnv2_116d_update01\)dft_3D_formation_energy --dataset jarvis "
    # "--figshare_target formation_energy_peratom --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01"
    # " --usePolynomial 3 --newEnvelope --useElectronegativity --normalizedElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2",

    # "python main.py --seed 210 --name 8five_seed\(210\)_\(128batch_eNeg+newRBF_3_Poly_newEnv2_116d_update01\)dft_3D_formation_energy --dataset jarvis "
    # "--figshare_target formation_energy_peratom --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01"
    # " --usePolynomial 3 --newEnvelope --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2",

    # "python main.py --seed 210 --name smoothEnv02_8five_seed\(210\)_\(128batch_eNeg+newRBF05_3_Poly_newEnv2_116d_update01\)dft_3D_formation_energy --dataset jarvis "
    # "--figshare_target formation_energy_peratom --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01"
    # " --usePolynomial 3 --newEnvelope --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2",

    # "python main.py --seed 204 --name seed\(204\)_\(norm_eNeg_3_Poly_newEnv2_116d_update01\)dft_3D_formation_energy --dataset jarvis "
    # "--figshare_target formation_energy_peratom --threads 6 --workers 3 --epochs 300 --atom_init atom_features\(116d\)_update01"
    # " --usePolynomial 3 --newEnvelope --useElectronegativity --normalizedElectronegativity",
    # ---- ⭐ useElectronegativity ⭐ ----
]


ig = [
    # 🚀 formation_energy
    # "python main.py --seed 205 --name 8five_seed(205)_(128batch_norm_eNeg_3_Poly_newEnv2_116d_update01)dft_3D_formation_energy --dataset jarvis"
    # " --figshare_target formation_energy_peratom --threads 10 --workers 5 --epochs 300 --atom_init atom_features(116d)_update01"
    # " --usePolynomial 3 --newEnvelope --useElectronegativity --normalizedElectronegativity --batch 128"
    # " --disableUpdateEdge --limitedUpdateEdge 2 --ig",

    "python main.py --seed 212 --name 8five_seed(212)_(128batch_eNeg+newRBF_3_Poly_newEnv2_116d_update01)dft_3D_formation_energy --dataset jarvis"
    " --figshare_target formation_energy_peratom --threads 10 --workers 5 --epochs 300 --atom_init atom_features(116d)_update01"
    " --usePolynomial 3 --newEnvelope --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --ig"

    # 🚀 mbj_bandgap
    # "python main.py --seed 293 --name 8five_seed\(293\)_\(128batch_norm_eNeg_3_Poly_newEnv2_116d_update01\)dft_3D_mbj_bandgap --dataset jarvis"
    # " --figshare_target mbj_bandgap --threads 6 --workers 3 --epochs 300 --atom_init atom_features\(116d\)_update01"
    # " --usePolynomial 3 --newEnvelope --useElectronegativity --normalizedElectronegativity --batch 128"
    # " --disableUpdateEdge --limitedUpdateEdge 2 --ig"

    # 🚀 opt_bandgap
    # "python main.py --seed 449 --name 8five_seed\(449\)_\(128batch_eNeg_3_Poly_newEnv2_116d_update01\)dft_3D_opt_bandgap --dataset jarvis "
    # "--figshare_target optb88vdw_bandgap --threads 6 --workers 3 --epochs 300 --atom_init atom_features\(116d\)_update01"
    # " --usePolynomial 3 --newEnvelope --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --ig",

    # 🚀 total_energy
    # "python main.py --seed 310 --name 8five_seed\(310\)_\(128batch_eNeg+newRBF05_3_Poly_newEnv2_116d_update01\)dft_3D_total_energy --dataset jarvis"
    # " --figshare_target optb88vdw_total_energy --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01"
    # " --usePolynomial 3 --newEnvelope --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --ig"

    # 🚀ehull
    # "python main.py --seed 501 --name 8five_seed\(501\)_\(128batch_eNeg+newRBF03_3_Poly_newEnv2_116d_update01\)dft_3D_ehull"
    # " --dataset jarvis --figshare_target ehull --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01"
    # " --usePolynomial 3 --newEnvelope --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --ig"

    # "python main.py --seed 505 --name 8five_seed\(505\)_\(128batch_eNeg+newRBF03_3_Poly_newEnv2_116d_update01\)dft_3D_ehull"
    # " --dataset jarvis --figshare_target ehull --threads 10 --workers 5 --epochs 400 --atom_init atom_features\(116d\)_update01"
    # " --usePolynomial 3 --newEnvelope --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --ig"

    # 🚀 e_form
    # "python main.py --seed 234 --name 8five_seed\(234\)_\(128batch_3_Poly_newEnv2_116d_update01\)megnet_formation_energy --dataset megnet "
    # "--figshare_target e_form --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01"
    # " --usePolynomial 3 --newEnvelope --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --ig",

    # "python main.py --seed 232 --name 8five_seed\(232\)_\(128batch_norm_eNeg_3_Poly_newEnv2_116d_update01\)megnet_formation_energy"
    # " --dataset megnet --figshare_target e_form --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01"
    # " --usePolynomial 3 --newEnvelope --useElectronegativity --normalizedElectronegativity --batch 128"
    # " --disableUpdateEdge --limitedUpdateEdge 2 --ig"

    # 🚀gap pbe
    # "python main.py --seed 334 --name 8five_seed\(334\)_\(500_128batch_norm_eNeg_3_Poly_newEnv2_116d_update01\)megnet_bandgap"
    # " --dataset megnet --threads 10 --workers 5 --epochs 500 --atom_init atom_features\(116d\)_update01"
    # " --usePolynomial 3 --newEnvelope --useElectronegativity --normalizedElectronegativity --batch 128"
    # " --disableUpdateEdge --limitedUpdateEdge 2 --ig"

    # "python main.py --seed 334 --name 8five_seed\(334\)_\(500_128batch_eNeg+newRBF_3_Poly_newEnv2_116d_update01\)megnet_bandgap"
    # " --dataset megnet  --threads 10 --workers 5 --epochs 500 --atom_init atom_features\(116d\)_update01"
    # " --usePolynomial 3 --newEnvelope --useElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --ig"

    # 🚀shear modulus


    # 🚀bulk modulus

]

for cmd in ig:
    os.system(cmd)
    print("--------------------------------------------------")
    print(f"Command '{cmd}' executed successfully.")
    # print("验证: 使用 自注意力 计算权重")
    print("--------------------------------------------------")
