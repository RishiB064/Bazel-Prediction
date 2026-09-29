import os
import subprocess
import json
import pandas as pd

# Author: Rishi Bharadwaj Ramesh
# Description: Reads InputData.csv, checks out commits, runs Bazel, and extracts CPUTime.

REPO_DIR = "bazel_repo"
INPUT_CSV = "InputData.csv"
OUTPUT_CSV = "build_metrics.csv"

def run_cmd(cmd, cwd=None):
    result = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout.strip()

def collect_cpu_times():
    # Load the existing input features
    df = pd.read_csv(INPUT_CSV)
    
    if not os.path.exists(REPO_DIR):
        print("Cloning Bazel repository...")
        run_cmd(f"git clone https://github.com/bazelbuild/bazel.git {REPO_DIR}")
    
    cpu_times = []
    
    # Note: To run a quick test before waiting hours, uncomment the line below:
    #df = df.head(3)
    
    print(f"Processing {len(df)} commits...")
    for index, row in df.iterrows():
        commit_id = row['ID']
        print(f"Building commit {index + 1}/{len(df)}: {commit_id}")
        
        run_cmd(f"git checkout {commit_id}", cwd=REPO_DIR)
        
        bep_file = f"bep_{commit_id}.json"
        run_cmd(f"bazel build //src:bazel-dev --build_event_json_file={bep_file}", cwd=REPO_DIR)
        
        cpu_time_ms = None
        bep_path = os.path.join(REPO_DIR, bep_file)
        
        if os.path.exists(bep_path):
            with open(bep_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        if "buildMetrics" in event and "timingMetrics" in event["buildMetrics"]:
                            cpu_time_ms = int(event["buildMetrics"]["timingMetrics"]["cpuTimeInMs"])
                            break
                    except json.JSONDecodeError:
                        continue
            os.remove(bep_path)
            
        cpu_times.append(cpu_time_ms)
        
    # Append target variable and drop any builds that failed to generate a BEP
    df['cpu_time'] = cpu_times
    df_clean = df.dropna(subset=['cpu_time'])
    
    df_clean.to_csv(OUTPUT_CSV, index=False)
    print(f"Dataset successfully saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    collect_cpu_times()