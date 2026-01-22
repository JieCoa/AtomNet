import os

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0, 1, 2, 3

# 切换到上一级目录
os.chdir("..")

jarvis = [
    # ----- jarvis -----
    # 🚀形成能
    # "python main.py --seed 205 --name dft_3D_formation_energy --dataset jarvis"
    # " --figshare_target formation_energy_peratom --threads 10 --workers 5 --epochs 300 --batch 128 --usePolynomial 0",
    
    # "python main.py --seed 205 --name dft_3D_formation_energy --dataset jarvis"
    # " --figshare_target formation_energy_peratom --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01"
    # " --batch 128 --disableUpdateEdge --limitedUpdateEdge 2",

    # ---- ⭐ useElectronegativity ⭐ ----
    # "python main.py --seed 212 --name dft_3D_formation_energy --dataset jarvis  --figshare_target formation_energy_peratom"
    # " --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity"
    # " --normalizedElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2",

    "python main.py --seed 206 --name dft_3D_formation_energy --dataset jarvis --figshare_target formation_energy_peratom"
    " --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity"
    " --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type simply --electronegativity_type newRBF",

    # ---- ⭐ useElectronegativity ⭐ ----


    # 🚀mbj_bandgap
    # "python main.py --seed 293 --name dft_3D_mbj_bandgap --dataset jarvis --figshare_target mbj_bandgap  --epochs 300"
    # " --atom_init cgcnn  --usePolynomial 0",

    # "python main.py --seed 296 --name dft_3D_mbj_bandgap --dataset jarvis --figshare_target mbj_bandgap --epochs 300"
    # " --atom_init atom_features\(116d\)_update01 --envelope_type simply",

    # ---- ⭐ useElectronegativity ⭐ ----
    # "python main.py --seed 291 --name dft_3D_mbj_bandgap --dataset jarvis --figshare_target mbj_bandgap --epochs 300"
    # " --atom_init atom_features\(116d\)_update01 --usePolynomial 3 --batch 64 --useElectronegativity --envelope_type simply",

    # "python main.py --seed 295 --name dft_3D_mbj_bandgap --dataset jarvis --figshare_target mbj_bandgap --epochs 300"
    # " --atom_init atom_features\(116d\)_update01 --usePolynomial 3 --batch 128 --useElectronegativity --normalizedElectronegativity",

    # ---- ⭐ useElectronegativity ⭐ ----


    # 🚀optb88vdw_bandgap
    # "python main.py --seed 448 --name dft_3D_opt_bandgap --dataset jarvis --figshare_target optb88vdw_bandgap"
    # " --epochs 300 --usePolynomial 0",

    # "python main.py --seed 448 --name dft_3D_opt_bandgap --dataset jarvis --figshare_target optb88vdw_bandgap"
    # " --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --batch 128"
    # " --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type cubic",

    # ---- ⭐ useElectronegativity ⭐ ----
    # "python main.py --seed 447 --name dft_3D_opt_bandgap --dataset jarvis --figshare_target optb88vdw_bandgap"
    # " --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity",

    # "python main.py --seed 449 --name dft_3D_opt_bandgap --dataset jarvis --figshare_target optb88vdw_bandgap"
    # " --epochs 300 --atom_init atom_features\(116d\)_update01 --usePolynomial 3 --useElectronegativity"
    # " --normalizedElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2",

    # "python main.py --seed 448 --name dft_3D_opt_bandgap --dataset jarvis --figshare_target optb88vdw_bandgap"
    # " --epochs 300 --atom_init atom_features\(116d\)_update01 --usePolynomial 3 --useElectronegativity --batch 128"
    # " --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type simply",

    # ---- ⭐ useElectronegativity ⭐ ----


    # 🚀optb88vdw_total_energy
    # "python main.py --seed 303 --name dft_3D_total_energy --dataset jarvis --figshare_target optb88vdw_total_energy"
    # --threads 10 --workers 5 --epochs 300 --usePolynomial 0",

    # "python main.py --seed 303 --name smoothEnv02_8five_seed\(303\)_\(128batch_3_Poly_newEnv2_116d_update01\)dft_3D_total_energy --dataset jarvis "
    # "--figshare_target optb88vdw_total_energy --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01"
    # " --usePolynomial 3 --batch 128 --disableUpdateEdge --limitedUpdateEdge 2",

    # ---- ⭐ useElectronegativity ⭐ ----
    # "python main.py --seed 310 --name dft_3D_total_energy --dataset jarvis --figshare_target optb88vdw_total_energy"
    # " --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity"
    # " --batch 128 --disableUpdateEdge --limitedUpdateEdge 2",

    # "python main.py --seed 303 --name dft_3D_total_energy --dataset jarvis --figshare_target optb88vdw_total_energy"
    # " --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity"
    # " --normalizedElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2",

    # "python main.py --seed 303 --name dft_3D_total_energy --dataset jarvis --figshare_target optb88vdw_total_energy"
    # " --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 128"
    # " --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type cubic --electronegativity_type newRBF",

    # ---- ⭐ useElectronegativity ⭐ ----

    # 🚀ehull
    # "python main.py --seed 502 --name dft_3D_ehull --dataset jarvis --figshare_target ehull --epochs 300"
    # " --usePolynomial 0",

    # "python main.py --seed 506 --name dft_3D_ehull --dataset jarvis --figshare_target ehull --threads 10 --workers 5"
    # " --epochs 300 --atom_init atom_features\(116d\)_update01 --batch 128 --disableUpdateEdge --limitedUpdateEdge 2",

    # ---- ⭐ useElectronegativity ⭐ ----
    # "python main.py --seed 501 --name dft_3D_ehull --dataset jarvis --figshare_target ehull --threads 10 --workers 5"
    # " --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --normalizedElectronegativity"
    # " --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type cubic",

    # "python main.py --seed 510 --name dft_3D_ehull --dataset jarvis --figshare_target ehull --threads 10 --workers 5"
    # " --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 128 --disableUpdateEdge"
    # " --limitedUpdateEdge 2 --envelope_type cubic --electronegativity_type newRBF03",

    # "python main.py --seed 510 --name dft_3D_ehull --dataset jarvis --figshare_target ehull --threads 10 --workers 5"
    # " --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 128 --disableUpdateEdge"
    # " --limitedUpdateEdge 2 --envelope_type cubic --electronegativity_type newRBF04",

    # ---- ⭐ useElectronegativity ⭐ ----
]

megnet = [
    # ----- megnet -----
    # 🚀e_form
    # "python main.py --seed 230 --name megnet_formation_energy --dataset megnet --figshare_target e_form --epochs 300",

    # "python main.py --seed 236 --name megnet_formation_energy --dataset megnet --figshare_target e_form --epochs 300"
    # " --atom_init atom_features\(116d\)_update01 --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type simply",

    # ---- ⭐ useElectronegativity ⭐ ----
    # "python main.py --seed 235 --name megnet_formation_energy --dataset megnet --figshare_target e_form --threads 10"
    # " --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 128"
    # " --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type simply --max_neighbours -1",

    # "python main.py --seed 235 --name megnet_formation_energy --dataset megnet --figshare_target e_form --threads 10"
    # " --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity"
    # " --normalizedElectronegativity --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type simply"
    # " --max_neighbours -1",

    # ---- ⭐ useElectronegativity ⭐ ----

    # 🚀gap pbe
    # "python main.py --seed 332 --name 8five_seed\(332\)_\(128batch_116d_update01\)megnet_bandgap --dataset megnet "
    # " --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --batch 128"
    # " --disableUpdateEdge --limitedUpdateEdge 2",

    # "python main.py --seed 331 --name megnet_bandgap --dataset megnet --threads 10 --workers 5 --epochs 300"
    # " --atom_init atom_features\(116d\)_update01 --batch 128 --disableUpdateEdge --limitedUpdateEdge 2"
    # " --envelope_type cubic --max_neighbours -1",

    # ---- ⭐ useElectronegativity ⭐ ----
    # "python main.py --seed 331 --name megnet_bandgap --dataset megnet --threads 10 --workers 5 --epochs 300"
    # " --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 128 --disableUpdateEdge"
    # " --limitedUpdateEdge 2 --envelope_type cubic --max_neighbours -1",

    # "python main.py --seed 331 --name megnet_bandgap --dataset megnet --threads 10 --workers 5 --epochs 300"
    # " --atom_init atom_features\(116d\)_update01 --useElectronegativity --normalizedElectronegativity --batch 128"
    # " --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type cubic --max_neighbours 25",

    # ---- ⭐ useElectronegativity ⭐ ----

    # 🚀shear modulus
    # "python main.py --seed 440 --name megnet_shear_modulus --dataset megnet --figshare_target 'shear modulus'"
    # " --epochs 300 --atom_init cgcnn",

    # "python main.py --seed 440 --name megnet_shear_modulus --dataset megnet --figshare_target 'shear modulus'"
    # " --epochs 300 --atom_init atom_features\(116d\)_update01 --batch 64 --disableUpdateEdge --limitedUpdateEdge 2"
    # " --envelope_type cubic",

    # ---- ⭐ useElectronegativity ⭐ ----
    # "python main.py --seed 440 --name megnet_shear_modulus --dataset megnet --figshare_target 'shear modulus'"
    # " --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 64"
    # " --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type cubic --electronegativity_type newRBF02",

    # "CUDA_VISIBLE_DEVICES=1 python main.py --seed 440 --name megnet_shear_modulus --dataset megnet --figshare_target 'shear modulus'"
    # " --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --normalizedElectronegativity"
    # " --batch 64 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type cubic",

    # ---- ⭐ useElectronegativity ⭐ ----

    # 🚀bulk modulus
    # "python main.py --seed 536 --name megnet_bulk_modulus --dataset megnet --figshare_target 'bulk modulus' --threads 10"
    # " --workers 5 --epochs 300 --atom_init cgcnn --batch 64",

    # "python main.py --seed 536 --name megnet_bulk_modulus --dataset megnet --figshare_target 'bulk modulus' --threads 10"
    # " --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --batch 64 --disableUpdateEdge"
    # " --limitedUpdateEdge 2 --envelope_type simply",

    # ---- ⭐ useElectronegativity ⭐ ----
    # "python main.py --seed 536 --name megnet_bulk_modulus --dataset megnet --figshare_target 'bulk modulus' --epochs 300"
    # " --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 128 --disableUpdateEdge"
    # " --limitedUpdateEdge 2 --envelope_type simply",

    # "python main.py --seed 536 --name megnet_bulk_modulus --dataset megnet --figshare_target 'bulk modulus' --epochs 300"
    # " --atom_init atom_features\(116d\)_update01 --useElectronegativity --normalizedElectronegativity --batch 64"
    # " --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type simply",

    # "python main.py --seed 536 --name megnet_bulk_modulus --dataset megnet --figshare_target 'bulk modulus' --epochs 300"
    # " --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 128 --disableUpdateEdge"
    # " --limitedUpdateEdge 2 --envelope_type cubic --electronegativity_type newRBF",

    # ---- ⭐ useElectronegativity ⭐ ----
]

ig = [
    # 🚀 formation_energy
    # "python main.py --seed 206 --name dft_3D_formation_energy --dataset jarvis --figshare_target formation_energy_peratom"
    # " --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity"
    # " --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type simply --electronegativity_type newRBF"
    # " --ig",

    # 🚀 mbj_bandgap
    # "python main.py --seed 296 --name dft_3D_mbj_bandgap --dataset jarvis --figshare_target mbj_bandgap --epochs 300"
    # " --atom_init atom_features\(116d\)_update01 --envelope_type simply --ig",

    # 🚀optb88vdw_bandgap
    # "python main.py --seed 448 --name dft_3D_opt_bandgap --dataset jarvis --figshare_target optb88vdw_bandgap"
    # " --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --batch 128"
    # " --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type cubic --ig",

    # 🚀optb88vdw_total_energy
    # "python main.py --seed 303 --name dft_3D_total_energy --dataset jarvis --figshare_target optb88vdw_total_energy"
    # " --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 128"
    # " --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type cubic --electronegativity_type newRBF05 --ig",

    # 🚀ehull
    # "python main.py --seed 510 --name dft_3D_ehull --dataset jarvis --figshare_target ehull --threads 10 --workers 5"
    # " --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 128 --disableUpdateEdge"
    # " --limitedUpdateEdge 2 --envelope_type cubic --electronegativity_type newRBF04 --ig",

    # 🚀e_form
    # "python main.py --seed 235 --name megnet_formation_energy --dataset megnet --figshare_target e_form --threads 10"
    # " --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 128"
    # " --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type simply --max_neighbours -1 --ig",

    # 🚀gap pbe
    # "python main.py --seed 331 --name megnet_bandgap --dataset megnet --threads 10 --workers 5 --epochs 300"
    # " --atom_init atom_features\(116d\)_update01 --batch 128 --disableUpdateEdge --limitedUpdateEdge 2"
    # " --envelope_type cubic --max_neighbours -1 --ig",

    # 🚀shear modulus
    # "python main.py --seed 440 --name megnet_shear_modulus --dataset megnet --figshare_target 'shear modulus'"
    # " --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 64"
    # " --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type cubic --electronegativity_type newRBF02 --ig",

    # 🚀bulk modulus
    # "python main.py --seed 536 --name megnet_bulk_modulus --dataset megnet --figshare_target 'bulk modulus' --epochs 300"
    # " --atom_init atom_features\(116d\)_update01 --useElectronegativity --normalizedElectronegativity --batch 64"
    # " --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type simply --ig",
]

inference = [
    # 🚀 formation_energy
    # "python main.py --seed 206 --name dft_3D_formation_energy --dataset jarvis --figshare_target formation_energy_peratom"
    # " --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity"
    # " --batch 128 --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type simply --electronegativity_type newRBF"
    # " --inference",

    # 🚀 mbj_bandgap
    # "python main.py --seed 296 --name dft_3D_mbj_bandgap --dataset jarvis --figshare_target mbj_bandgap --epochs 300"
    # " --atom_init atom_features\(116d\)_update01 --envelope_type simply --inference",

    # 🚀optb88vdw_bandgap
    # "python main.py --seed 448 --name dft_3D_opt_bandgap --dataset jarvis --figshare_target optb88vdw_bandgap"
    # " --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --batch 128"
    # " --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type cubic --inference",

    # 🚀optb88vdw_total_energy
    # "python main.py --seed 303 --name dft_3D_total_energy --dataset jarvis --figshare_target optb88vdw_total_energy"
    # " --threads 10 --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 128"
    # " --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type cubic --electronegativity_type newRBF05 --inference",

    # 🚀ehull
    # "python main.py --seed 510 --name dft_3D_ehull --dataset jarvis --figshare_target ehull --threads 10 --workers 5"
    # " --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 128 --disableUpdateEdge"
    # " --limitedUpdateEdge 2 --envelope_type cubic --electronegativity_type newRBF04 --inference",

    # 🚀e_form
    # "python main.py --seed 235 --name megnet_formation_energy --dataset megnet --figshare_target e_form --threads 10"
    # " --workers 5 --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 128"
    # " --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type simply --max_neighbours -1 --inference",

    # 🚀gap pbe
    # "python main.py --seed 331 --name megnet_bandgap --dataset megnet --threads 10 --workers 5 --epochs 300"
    # " --atom_init atom_features\(116d\)_update01 --batch 128 --disableUpdateEdge --limitedUpdateEdge 2"
    # " --envelope_type cubic --max_neighbours -1 --inference",

    # 🚀shear modulus
    # "python main.py --seed 440 --name megnet_shear_modulus --dataset megnet --figshare_target 'shear modulus'"
    # " --epochs 300 --atom_init atom_features\(116d\)_update01 --useElectronegativity --batch 64"
    # " --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type cubic --electronegativity_type newRBF02 --inference",

    # 🚀bulk modulus
    # "python main.py --seed 536 --name megnet_bulk_modulus --dataset megnet --figshare_target 'bulk modulus' --epochs 300"
    # " --atom_init atom_features\(116d\)_update01 --useElectronegativity --normalizedElectronegativity --batch 64"
    # " --disableUpdateEdge --limitedUpdateEdge 2 --envelope_type simply --inference",
]

for cmd in jarvis:
    os.system(cmd)
    print("--------------------------------------------------")
    print(f"Command '{cmd}' executed successfully.")
    # print("验证: 使用 自注意力 计算权重")
    print("--------------------------------------------------")
