# Predicting Bazel Build Times

**Authors:** Rishi Bharadwaj Ramesh Raghav Budur

## Overview
The project implements a machine learning model to predict the CPU-time of Bazel builds. It utilizes a linear regression algorithm with feature crossing to capture non-linear relationships between the inputs (the most common file path prefix and file type in a commit) and the build execution time. This predictive model allows developers to optimize resource usage in advance of executing a build.

## Prerequisites
Ensure the following tools are installed and configured in your Windows environment:
* Python 3.8 or higher
* Git
* Bazel (added to your Windows PATH)
* Windows PowerShell

## Setup Instructions
The following instructions detail how to set up the project environment.

1. Clone this repository to your local machine:
   ```powershell
   git clone [https://github.com/RishiB064/Bazel-Build-Times.git](https://github.com/RishiB064/Bazel-Build-Times.git)
   cd Bazel-Build-Times
   ```

2. Install all the required dependencies
   ```powershell
   pip install tensorflow pandas scikit-learn
   ```

## Running the Project
The following instructions detail how to run the data collection and model training scripts[cite: 1].

1. Data Collection
   The data collection script clones the open-source bazelbuild/bazel repository, executes a git diff across 500 commits, and extracts the target CPU-time from the Build Event Protocol (BEP) JSON files[cite: 2].

Run the extraction script in PowerShell:

```powershell
python data_collection.py
```

2. Model Training and Evaluation

   Once the dataset is generated, run the prediction script. This script preprocesses the string features into discrete integers using bucketization, applies feature crossing, and trains a linear regression model optimized with stochastic gradient descent[cite: 2].

Run the training script in PowerShell:

```powershell
python predict_build_time.py
```

The console will output the Validation Loss, Testing Loss, and the final Root Mean Squared Error (RMSE) in milliseconds[cite: 2].
