#!/usr/bin/python3
# author Caojie
# 2025年03月10日23点57分

from jarvis.db.figshare import data as jdata
import json
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示为方块的问题


def printJarvisDataStructure(jarvis_data: list):
    """
    打印数据集中的第一条数据
    """
    if jarvis_data:
        first_entry = jarvis_data[0]
        print("dft_3D 的类型：", type(jarvis_data))
        print("dft_3D 数据集中的第一条数据：")
        print(json.dumps(first_entry, indent=4))  # 使用 json.dumps 格式化输出
        element = first_entry.get("atoms")["elements"]
        print("拥有元素：", element)
    else:
        print("未找到 dft_3D_2021 数据集。")


def printMegnetDataStructure(megnet_data: list):
    """
    打印数据集中的第一条数据
    """
    if megnet_data:
        first_entry = megnet_data[0]
        print("megnet_data 的类型：", type(megnet_data))
        print("MEGNet 数据集中的第一条数据：")
        print(json.dumps(first_entry, indent=4))  # 使用 json.dumps 格式化输出
        element = first_entry.get("atoms")["elements"]
        print("拥有元素：", element)
    else:
        print("未找到 MEGNet 数据集。")


def getIdByElement(dataset_name, datasets) -> list:
    """
    根据元素筛选材料

    params:
        dataset_name: 数据集名称
        datasets: 数据集
    """
    target_elememt = {"Fe", "Co", "Ni"}

    material_ids = []

    for ma in datasets:
        # 获取当前数据的 elements 列表
        elements = ma.get("atoms")["elements"]   # 若不存在，返回空列表
        # 检查 elements 是否包含 Fe, Co, Ni 中的任意一个
        if any(elem in target_elememt for elem in elements):
            if dataset_name == "megnet":
                material_ids.append(ma.get("id"))
            elif dataset_name == "dft_3d_2021":
                material_ids.append(ma.get("reference"))
            else:
                print("未知数据集")

    if material_ids:
        # print("找到包含 Fe, Co 或 Ni 的数据的 ID 列表：")
        # print(json.dumps(material_ids, indent=4))  # indent 表示每个层级的 JSON 数据会缩进 4 个空格
        print("全部数据数量：", len(material_ids))
    else:
        print("未找到包含 Fe, Co 或 Ni 的数据。")

    return material_ids


def getDataById(dataset_name: str) -> list:
    """
    根据材料 ID 获取数据
    dft_3d_2021中，对应 Materaial Project 中 id 的键是 "reference"

    params:
        dataset_name: 数据集名称
    """
    megnet_data = jdata(dataset_name)

    data_list = []
    material_id_list = getIdByElement(dataset_name, megnet_data)

    for ma in megnet_data:
        if dataset_name == "megnet":
            if ma.get("id") in material_id_list:
                # print("找到 ID 为", ma.get("id"), "的数据：")
                data_list.append(ma)
        elif dataset_name == "dft_3d_2021":
            if ma.get("reference") in material_id_list:
                # print("找到 ID 为", ma.get("id"), "的数据：")
                data_list.append(ma)
        else:
            print("未知数据集")
    return data_list


def draw_histogram_LatticeData(data_list: list):
    lattice_list = []
    count = 0
    for material in tqdm(data_list):  # dict
        lattice_abc = material.get("atoms")["abc"]
        if len(lattice_abc) == 3 and lattice_abc != "na" and lattice_abc[0] != "" and lattice_abc[1] != "" and lattice_abc[2] != "":
            lattice_list.extend(lattice_abc)
        else:
            count += 1
    print(f'统计结果：{len(lattice_list) // 3} 个材料数据，{len(lattice_list)} 个晶格常数。')
    print(f"异常数据 {count} 个。")

    # 绘制直方图
    counts, bins, patches = plt.hist(lattice_list, bins=20, color="skyblue", edgecolor="black", alpha=0.7)

    # 在每个柱子上显示数量
    for count, x in zip(counts, bins):
        if count > 0:  # 避免在空柱子上显示
            plt.text(x + (bins[1] - bins[0]) / 2, count, str(int(count)),
                     ha='center', va='bottom')

    # 添加局部放大图
    ax = plt.gca()
    inset_ax = ax.inset_axes([0.25, 0.25, 0.7, 0.65])  # [x位置, y位置, 宽, 高]
    sub_counts, sub_bins, sub_patches = inset_ax.hist([x for x in lattice_list if 0 <= x <= 20], bins=20, color="skyblue", edgecolor="black")
    for count, left, right in zip(sub_counts, sub_bins[:-1], sub_bins[1:]):
        if count > 0:
            inset_ax.text(
                (left + right) / 2,  # 柱子中心位置
                count,  # 纵坐标（高度）
                str(int(count)),  # 转成整数
                ha='center', va='bottom', fontsize=8
            )
    inset_ax.set_title("局部分布 [0, 20]", fontsize=10)

    # 添加标签
    plt.xlabel("数值区间")
    plt.ylabel("频数")
    plt.title("完整直方图 + 局部分布")
    plt.show()


if __name__ == "__main__":
    # 获取 MEGNet 数据集
    # megnet_data = jdata("megnet")  # megnet

    dft_3D = jdata("dft_3d_2021")
    print("数据加载完成！")

    # printMegnetDataStructure(megnet_data)

    printJarvisDataStructure(dft_3D)  # 打印 dft_3D 的第一个数据信息

    # 统计 lattice 数据
    # draw_histogram_LatticeData(dft_3D)

    # data = getDataById("dft_3d_2021")
    # print(json.dumps(data[0], indent=4))  # 检测 getDataById() 是否正确运行
