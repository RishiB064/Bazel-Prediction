import os
import subprocess


# Author: Rishi Bharadwaj 

import pandas as pd

REPO_DIR = "bazel_repo"


def run_cmd(cmd, cwd=None):
    r = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, errors="replace")
    return r.returncode, (r.stdout or "")


def main():
    inputs = pd.read_csv("InputData.csv")
    times = pd.read_csv("CPUTimes.csv")
    inputs.columns = [str(c).strip().lower() for c in inputs.columns]
    times.columns = [str(c).strip().lower() for c in times.columns]

    if "cpu_time" not in times.columns:
        for alt in ("cputime", "cpu_time_ms", "cputimems"):
            if alt in times.columns:
                times = times.rename(columns={alt: "cpu_time"})
                break
    if "cpu_time" not in times.columns:
        raise SystemExit(f"No CPU-time column in CPUTimes.csv ({list(times.columns)})")

    if "id" in inputs.columns and "id" in times.columns:
        merged = inputs.merge(times[["id", "cpu_time"]].drop_duplicates("id"),
                              on="id", how="left")
        matched = merged["cpu_time"].notna().sum()
        print(f"Merged on 'id': {matched}/{len(merged)} rows matched")
    elif len(inputs) == len(times):
        print("[warn] no shared 'id' column -> falling back to positional merge")
        merged = inputs.copy()
        merged["cpu_time"] = times["cpu_time"].values
    else:
        raise SystemExit("Cannot align files: no shared 'id' column and row counts differ")

    merged = merged.rename(columns={"id": "ID"})

    # ---- free extra features from the local clone (no bazel re-runs needed) ----
    if "ID" in merged.columns and os.path.isdir(os.path.join(REPO_DIR, ".git")):
        rc, out = run_cmd('git log --all --format="%H %at"', cwd=REPO_DIR)
        ts = {}
        for line in out.splitlines():
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    ts[parts[0]] = int(parts[1])
                except ValueError:
                    pass
        merged["commit_time"] = merged["ID"].astype(str).map(ts)
        rate = merged["commit_time"].notna().mean() * 100
        if rate >= 50:
            merged["commit_index"] = merged["commit_time"].rank(method="first")
            print(f"Enriched with git features (matched {rate:.0f}% of commits)")
        else:
            merged = merged.drop(columns=["commit_time"])
            print(f"[info] git enrichment skipped (only {rate:.0f}% commits matched)")

    before = len(merged)
    merged = merged.dropna(subset=["cpu_time"]).drop_duplicates(subset=["ID"], keep="last")
    if len(merged) != before:
        print(f"Dropped {before - len(merged)} rows (missing cpu_time or duplicate ID)")

    merged.to_csv("build_metrics.csv", index=False)
    print(f"\nSaved build_metrics.csv  shape={merged.shape}")
    print(merged["cpu_time"].describe().to_string())


if __name__ == "__main__":
    main()
