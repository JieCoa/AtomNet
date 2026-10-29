"""
推理输入的归一化：

把「jarvis 风格 json / 单个 CIF / CIF 目录」统一成 Figshare_Dataset 所需的
`[{"id": ..., "atoms": {...}}, ...]` 记录列表，其中 atoms 为 jarvis.core.atoms.Atoms 字典。
"""

import json
import logging
from pathlib import Path

CIF_SUFFIXES = (".cif", ".mcif")
JSON_SUFFIXES = (".json",)
_REQUIRED_ATOMS_KEYS = ("lattice_mat", "coords", "elements")
_ID_KEYS = ("id", "reference", "jid", "mp_id")


def load_inference_records(source, primitive=False):
    """将推理输入归一化为记录列表，每条记录包含 "id" 与 "atoms"。

    source 支持三种形态：
      - json 文件：`{"mp-19306": {"atoms": {...}}, ...}` 的字典形式（键即 id），
        同时兼容旧的列表形式 `[{"atoms": {...}}, ...]`，此时 id 取条目里的
        id/reference/jid，缺省为 `文件名#序号`
      - 单个 cif 文件
      - 目录：批量读取目录下的 cif 文件（按文件名排序）；目录内没有 cif 时，
        若恰好只有一个 json 文件则按 json 处理
    """
    path = Path(source)
    if path.is_dir():
        return _records_from_dir(path, primitive)
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in CIF_SUFFIXES:
            return [{"id": path.stem, "atoms": cif_to_atoms_dict(path, primitive=primitive)}]
        if suffix in JSON_SUFFIXES:
            return _records_from_json(path)
        raise ValueError(f"不支持的推理输入后缀：{path.suffix}（支持 .json、.cif 或包含 cif 的目录）")
    raise FileNotFoundError(f"推理输入路径不存在：{path}")


def _records_from_dir(directory, primitive):
    cif_files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in CIF_SUFFIXES)
    if cif_files:
        logging.info(f"从目录 {directory} 读取到 {len(cif_files)} 个 CIF 文件")
        return [{"id": p.stem, "atoms": cif_to_atoms_dict(p, primitive=primitive)} for p in cif_files]

    json_files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in JSON_SUFFIXES)
    if len(json_files) == 1:
        logging.info(f"目录 {directory} 中没有 CIF 文件，改用 {json_files[0].name}")
        return _records_from_json(json_files[0])
    if json_files:
        raise ValueError(f"目录 {directory} 中没有 CIF 文件，但存在多个 json，请显式指定其中某一个")
    raise FileNotFoundError(f"目录 {directory} 中没有 CIF 或 json 文件")


def _records_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):  # 兼容旧格式：[{"atoms": {...}}, ...]
        if not raw:
            raise ValueError(f"{path} 应包含非空的晶体结构列表")
        return [_record_from_entry(entry, f"{path.name}[{index}]", _entry_id(entry, path, index))
                for index, entry in enumerate(raw)]

    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path} 应包含非空的结构字典或结构列表")

    if _is_single_entry(raw):
        return [_record_from_entry(raw, path.name, _entry_id(raw, path, 0))]

    # 主格式：{"mp-19306": {"atoms": {...}}, ...}，键即 id
    return [_record_from_entry(entry, f"{path.name}['{key}']", str(key)) for key, entry in raw.items()]


def _is_single_entry(raw):
    """区分「单条结构」与「id -> 结构」的字典：前者的值里带 lattice_mat/elements。"""
    nested = raw.get("atoms")
    return _is_atoms_dict(nested) or _is_atoms_dict(raw)


def _is_atoms_dict(value):
    return isinstance(value, dict) and "lattice_mat" in value and "elements" in value


def _record_from_entry(entry, origin, record_id):
    if not isinstance(entry, dict):
        raise ValueError(f"{origin} 不是字典")
    atoms = entry.get("atoms", entry)
    if not isinstance(atoms, dict):
        raise ValueError(f"{origin} 缺少 atoms 字典")
    return {"id": record_id, "atoms": _normalize_atoms_dict(atoms, origin)}


def _entry_id(entry, path, index):
    for key in _ID_KEYS:
        value = entry.get(key)
        if value:
            return str(value)
    return f"{path.stem}#{index}"


def _normalize_atoms_dict(atoms, origin):
    missing = [key for key in _REQUIRED_ATOMS_KEYS if key not in atoms]
    if missing:
        raise ValueError(f"{origin} 缺少必需字段：{missing}")

    num_atoms = len(atoms["elements"])
    if len(atoms["lattice_mat"]) != 3 or any(len(row) != 3 for row in atoms["lattice_mat"]):
        raise ValueError(f"{origin} 的 lattice_mat 不是 3x3 矩阵")
    if len(atoms["coords"]) != num_atoms:
        raise ValueError(f"{origin} 的 coords（{len(atoms['coords'])}）与 elements（{num_atoms}）数量不一致")

    normalized = dict(atoms)
    normalized["cartesian"] = bool(atoms.get("cartesian", False))
    if not normalized.get("props") or len(normalized["props"]) != num_atoms:
        normalized["props"] = [""] * num_atoms
    return normalized


def cif_to_atoms_dict(cif_path, primitive=False):
    """把 CIF 文件转成 jarvis Atoms 字典（分数坐标 + 行向量晶格矩阵）。

    优先使用 pymatgen（可正确处理对称性展开与占位率），缺失或解析失败时回退到
    jarvis.core.atoms.Atoms.from_cif。primitive=True 时压缩为原胞，可与 jarvis
    数据集中的原胞表示对齐。
    """
    cif_path = Path(cif_path)

    try:
        from pymatgen.core import Structure
    except ImportError:
        logging.warning("未安装 pymatgen，回退到 jarvis.core.atoms.Atoms.from_cif 解析 CIF")
        return _atoms_dict_with_jarvis(cif_path)

    try:
        structure = Structure.from_file(str(cif_path))
    except Exception as exc:
        logging.warning(f"pymatgen 解析 {cif_path.name} 失败（{exc}），回退到 jarvis 解析")
        return _atoms_dict_with_jarvis(cif_path)

    if primitive:
        structure = structure.get_primitive_structure()
    return _structure_to_atoms_dict(structure, cif_path.name)


def _structure_to_atoms_dict(structure, origin):
    symbols = []
    for site in structure:
        species = site.species
        if len(species) == 1:
            symbols.append(species.elements[0].symbol)
            continue
        dominant = max(species.items(), key=lambda item: item[1])[0]
        logging.warning(
            f"{origin}: 位点 {[round(x, 4) for x in site.frac_coords]} 为无序/部分占位 {species}，取主元素 {dominant.symbol}"
        )
        symbols.append(dominant.symbol)

    # pymatgen 与 jarvis 的行向量约定一致：cart = frac @ lattice_mat
    return {
        "lattice_mat": structure.lattice.matrix.tolist(),
        "coords": structure.frac_coords.tolist(),
        "elements": symbols,
        "abc": list(structure.lattice.abc),
        "angles": list(structure.lattice.angles),
        "cartesian": False,
        "props": [""] * len(symbols),
    }


def _atoms_dict_with_jarvis(cif_path):
    from jarvis.core.atoms import Atoms as JarvisAtoms

    try:
        atoms = JarvisAtoms.from_cif(str(cif_path))
    except Exception as exc:
        raise RuntimeError(f"无法解析 CIF 文件 {cif_path}：{exc}") from exc
    if atoms is None:
        raise RuntimeError(f"无法解析 CIF 文件 {cif_path}：jarvis 未返回结构")
    return _normalize_atoms_dict(atoms.to_dict(), str(cif_path))
