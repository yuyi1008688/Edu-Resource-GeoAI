# -*- coding: utf-8 -*-
"""对比两个 model_report / summary json。"""
import argparse
import json


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, p + k + "."))
        else:
            out[p + k] = v
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--new", required=True)
ap.add_argument("--ref", required=True)
a = ap.parse_args()
fn = flat(json.load(open(a.new, encoding="utf-8")))
fr = flat(json.load(open(a.ref, encoding="utf-8")))
print(f"{'指标':44}{'本次':>16}{'基准':>16}")
print("-" * 76)
for k in sorted(set(fn) | set(fr)):
    x, y = fn.get(k, "-"), fr.get(k, "-")
    if isinstance(x, float):
        x = round(x, 4)
    if isinstance(y, float):
        y = round(y, 4)
    if isinstance(x, (list, tuple)):
        x = str(x)[:15]
    if isinstance(y, (list, tuple)):
        y = str(y)[:15]
    print(f"{k:44}{str(x):>16}{str(y):>16}")
