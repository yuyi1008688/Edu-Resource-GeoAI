#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""compare_chain.py — 4.4/4.5/4.7/5.0 复现输出与历史基准逐列对比工具。"""
import argparse
import numpy as np
import pandas as pd


def compare(new_path: str, ref_path: str, key: str = "school_id") -> None:
    new = pd.read_csv(new_path, encoding="utf-8-sig")
    ref = pd.read_csv(ref_path, encoding="utf-8-sig")
    print(f"新输出: {new.shape}  基准: {ref.shape}")
    print("新增列:", sorted(set(new.columns) - set(ref.columns)))
    print("缺失列:", sorted(set(ref.columns) - set(new.columns)))
    if key not in new.columns or key not in ref.columns:
        print(f"!! 无对齐键 {key}，改为按行序对比")
        m = pd.concat([new.reset_index(), ref.reset_index()], axis=1)
        align_n = min(len(new), len(ref))
    else:
        m = new.merge(ref, on=key, suffixes=("_new", "_ref"))
        align_n = len(m)
        print(f"按 {key} 对齐: {align_n} 行")
    common = [c for c in ref.columns if c in new.columns and c != key]
    print(f"\n{'列':26}{'最大绝对差':>14}{'中位相对差%':>12}{'相关系数':>10}")
    print("-" * 64)
    for c in common:
        cn, cr = c + "_new", c + "_ref"
        if cn not in m or cr not in m:
            continue
        a, b = m[cn], m[cr]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            d = (a - b).abs()
            denom = b.abs().replace(0, np.nan)
            rel = (d / denom * 100)
            corr = a.corr(b) if a.std() > 0 and b.std() > 0 else np.nan
            print(f"{c:26}{d.max():>14.4g}{rel.median():>12.3f}{corr:>10.4f}")
        else:
            eq = (a.astype(str) == b.astype(str)).mean()
            print(f"{c:26}{'类别一致率':>14}{eq*100:>11.1f}%{'-':>10}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--key", default="school_id")
    a = ap.parse_args()
    compare(a.new, a.ref, a.key)
