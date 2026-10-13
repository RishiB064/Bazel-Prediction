import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess

import pandas as pd


# Author: Rishi Bharadwaj 

REPO_DIR = "bazel_repo"
INPUT_CSV = "InputData.csv"
OUTPUT_CSV = "build_metrics.csv"
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def run_cmd(cmd, cwd=None, timeout=None):
    try:
        if isinstance(cmd, (list, tuple)):
            r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True,
                               errors="replace", timeout=timeout)
        else:
            r = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True,
                               errors="replace", timeout=timeout)
        return r.returncode, (r.stdout or "")
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def ensure_repo():
    if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
        print("Cloning Bazel repository...")
        run_cmd(f"git clone https://github.com/bazelbuild/bazel.git {REPO_DIR}")


def commit_time(commit):
    rc, out = run_cmd(["git", "show", "-s", "--format=%at", commit], cwd=REPO_DIR)
    lines = out.strip().splitlines()
    if rc == 0 and lines:
        try:
            return int(lines[-1].strip())
        except ValueError:
            pass
    return None


def diff_features(commit):
    rc, out = run_cmd(["git", "diff", "--numstat", f"{commit}~1", commit], cwd=REPO_DIR)
    if rc != 0:
        rc, out = run_cmd(["git", "diff", "--numstat", EMPTY_TREE, commit], cwd=REPO_DIR)

    files = ins = dele = 0
    exts, prefixes, kinds = Counter(), Counter(), Counter()
    for line in out.splitlines():
        p = line.split("\t")
        if len(p) < 3:
            continue
        a, b, name = p[0], p[1], p[2]
        try:
            ins += int(a)
        except ValueError:
            pass
        try:
            dele += int(b)
        except ValueError:
            pass
        files += 1
        ext = Path(name).suffix.lower().lstrip(".") or "none"
        exts[ext] += 1
        prefixes["/".join(name.replace("\\", "/").split("/")[:2])] += 1
        low = name.lower()
        if "third_party" in low:
            kinds["third_party"] += 1
        elif "/test" in low or low.startswith("test"):
            kinds["test"] += 1
        elif ext in ("md", "txt", "rst"):
            kinds["docs"] += 1
        else:
            kinds["src"] += 1
    total = max(files, 1)
    return {
        "num_files": files,
        "insertions": ins,
        "deletions": dele,
        "src_ratio": round(kinds["src"] / total, 4),
        "test_ratio": round(kinds["test"] / total, 4),
        "docs_ratio": round(kinds["docs"] / total, 4),
        "third_party_ratio": round(kinds["third_party"] / total, 4),
        "unique_ext": len(exts),
    }


def parse_bep_cpu(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tm = (ev.get("buildMetrics") or {}).get("timingMetrics") or {}
                v = tm.get("cpuTimeInMs")
                if v is not None:
                    try:
                        return int(v)
                    except (TypeError, ValueError):
                        pass
    except OSError:
        pass
    return None


def features_only(refresh=False):
    out_path = Path(OUTPUT_CSV)
    if not out_path.exists():
        raise SystemExit("build_metrics.csv not found; run full collection first")
    bm = pd.read_csv(out_path)
    if "ID" not in bm.columns:
        raise SystemExit(f"build_metrics.csv has no 'ID' column: {list(bm.columns)}")
    bm["ID"] = bm["ID"].astype(str)

    reason = None
    if refresh or "num_files" not in bm.columns:
        reason = "refresh requested" if refresh else "num_files column missing"
    else:
        nf = pd.to_numeric(bm["num_files"], errors="coerce")
        if nf.fillna(0).eq(0).all():
            reason = "existing num_files are all zero (v3 Windows '^' bug)"
        elif nf.median() > 1000:
            reason = "existing num_files are whole-tree sized (v3 Windows '^' bug)"

    if reason:
        print(f"[info] recomputing ALL rows: {reason}")
        todo = bm["ID"].tolist()
    else:
        todo = bm.loc[pd.to_numeric(bm["num_files"], errors="coerce").isna(), "ID"].tolist()

    todo = [c for c in todo if c and c.lower() not in ("nan", "none")]
    print(f"Computing git features for {len(todo)} commits (no bazel builds)...")
    if not todo:
        print("Nothing to do.")
        return

    rows = []
    for i, c in enumerate(todo, 1):
        f = {"ID": c, "commit_time": commit_time(c)}
        f.update(diff_features(c))
        rows.append(f)
        if i % 50 == 0:
            print(f"  {i}/{len(todo)}")

    fdf = pd.DataFrame(rows)
    bm = bm.drop(columns=[c for c in fdf.columns if c != "ID" and c in bm.columns])
    bm = bm.merge(fdf, on="ID", how="left")
    bm.to_csv(out_path, index=False)
    print(f"\nUpdated {out_path}: {bm.shape}")
    print("\nFeature sanity summary:")
    print(bm[["num_files", "insertions", "deletions", "unique_ext"]].describe().to_string())


def collect(args, df):
    out_path = Path(OUTPUT_CSV)
    done = {}
    if out_path.exists():
        prev = pd.read_csv(out_path)
        if {"ID", "cpu_time"}.issubset(prev.columns):
            ok = prev[prev["cpu_time"].notna()]
            done = dict(zip(ok["ID"].astype(str), ok["cpu_time"]))

    todo = [c for c in df["ID"].astype(str) if c not in done]
    if args.limit:
        todo = todo[: args.limit]

    if not todo:
        print("Every commit already has cpu_time. Use --features-only for diff metrics.")
        return

    records = []
    for i, commit in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {commit[:12]}")
        run_cmd(f"git checkout -q -f {commit}", cwd=REPO_DIR)

        rec = {"ID": commit, "commit_time": commit_time(commit)}
        rec.update(diff_features(commit))

        bep_name = f"bep_{commit}.json"
        rc, _ = run_cmd(
            f"bazel build //src:bazel-dev --noshow_progress "
            f"--build_event_json_file=../{bep_name}",
            cwd=REPO_DIR, timeout=args.timeout)
        rec["cpu_time"] = parse_bep_cpu(Path(bep_name)) if rc == 0 else None
        if Path(bep_name).exists():
            Path(bep_name).unlink()
        records.append(rec)

    new_df = pd.DataFrame(records)
    meta_cols = [c for c in ("prefix", "type") if c in df.columns]
    if meta_cols and len(new_df):
        new_df = new_df.merge(df[["ID"] + meta_cols].astype({"ID": str}), on="ID", how="left")

    if out_path.exists() and done:
        prev = pd.read_csv(out_path)
        prev = prev[prev["cpu_time"].notna()]
        for c in new_df.columns:
            if c not in prev.columns:
                prev[c] = pd.NA
        combined = pd.concat([prev, new_df], ignore_index=True)
    else:
        combined = new_df

    front = [c for c in ("ID", "prefix", "type") if c in combined.columns]
    combined = combined[front + [c for c in combined.columns if c not in front]]
    combined.to_csv(out_path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--features-only", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    ensure_repo()

    if args.features_only:
        features_only(refresh=args.refresh)
        return

    df = pd.read_csv(INPUT_CSV)
    if "ID" not in df.columns:
        raise SystemExit("InputData.csv must contain an 'ID' column")
    df["ID"] = df["ID"].astype(str)
    collect(args, df)


if __name__ == "__main__":
    main()
