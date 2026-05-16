import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import argparse

def plot_km_curves(opt):
    # 1. Path to the prediction file
    # By default, train_cv.py saves test predictions as:
    # {checkpoints_dir}/{exp_name}/{model_name}/{model_name}_{split}_pred_test.pkl
    
    # We will try to find the prediction file for Split 1 as a sample
    split_idx = 1
    pred_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, f"{opt.model_name}_{split_idx}_pred_test.pkl")
    
    if not os.path.exists(pred_path):
        # Fallback for earlier versions or patch-based runs
        pred_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, f"{opt.model_name}_{split_idx}_patch_pred_test.pkl")
    
    if not os.path.exists(pred_path):
        print(f"Error: Could not find prediction file at {pred_path}")
        print("Please check your --checkpoints_dir and --exp_name.")
        return

    print(f"Loading predictions from: {pred_path}")
    with open(pred_path, 'rb') as f:
        pred_data = pickle.load(f)

    # Structure of pred_test: [hazards, surv_times, censors, probs, ground_truth]
    hazards = pred_data[0]
    surv_times = pred_data[1]
    censors = pred_data[2]

    # 2. Split into High vs Low Risk groups based on median hazard
    median_risk = np.median(hazards)
    high_risk_idx = hazards > median_risk
    low_risk_idx = hazards <= median_risk

    # 3. Initialize Kaplan-Meier Fitter
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()

    plt.figure(figsize=(10, 7))

    # 4. Fit and Plot
    kmf_high.fit(surv_times[high_risk_idx], event_observed=censors[high_risk_idx], label=f'High Risk (n={sum(high_risk_idx)})')
    ax = kmf_high.plot_survival_function(color='red', linestyle='--')

    kmf_low.fit(surv_times[low_risk_idx], event_observed=censors[low_risk_idx], label=f'Low Risk (n={sum(low_risk_idx)})')
    kmf_low.plot_survival_function(ax=ax, color='blue')

    # 5. Log-Rank Test for statistical significance
    results = logrank_test(surv_times[high_risk_idx], surv_times[low_risk_idx], 
                           event_observed_A=censors[high_risk_idx], event_observed_B=censors[low_risk_idx])
    
    p_value = results.p_value

    # Formatting the plot
    plt.title(f'Kaplan-Meier Survival Curves (Split {split_idx})\nLog-rank p-value: {p_value:.2e}', fontsize=15)
    plt.xlabel('Time (Months)', fontsize=12)
    plt.ylabel('Survival Probability', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()

    # Save and show
    save_fig_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, f'km_curve_split_{split_idx}.png')
    plt.savefig(save_fig_path)
    print(f"Plot saved to: {save_fig_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/TCGA_GBMLGG', help='models are saved here')
    parser.add_argument('--exp_name', type=str, default='pathomic_crossattn', help='name of the experiment')
    parser.add_argument('--model_name', type=str, default='omic', help='model_name subfolder')
    
    opt = parser.parse_known_args()[0]
    plot_km_curves(opt)
