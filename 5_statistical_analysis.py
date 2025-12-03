#!/usr/bin/env python3
"""
Step 5: Statistical Analysis - A/B Testing
Performs t-test to prove statistical significance
Calculates p-value to validate the improvement
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
from datetime import datetime

def perform_ab_test(eval_results_file='evaluation_results.csv'):
    """Perform A/B test between baseline and agent systems"""
    
    if not os.path.exists(eval_results_file):
        print(f"❌ {eval_results_file} not found. Run: python 4_evaluation.py")
        return
    
    eval_df = pd.read_csv(eval_results_file)
    
    # Split by system
    baseline_scores = eval_df[eval_df['system'] == 'baseline_rag']['score'].values
    agent_scores = eval_df[eval_df['system'] == 'langgraph_agent']['score'].values
    
    print("=" * 80)
    print("📊 A/B STATISTICAL TEST ANALYSIS")
    print("=" * 80)
    
    # Descriptive Statistics
    print(f"\n🔴 BASELINE RAG SYSTEM:")
    print(f"   Sample Size: {len(baseline_scores)}")
    print(f"   Mean Score: {baseline_scores.mean():.3f}")
    print(f"   Std Dev: {baseline_scores.std():.3f}")
    print(f"   Min Score: {baseline_scores.min():.3f}")
    print(f"   Max Score: {baseline_scores.max():.3f}")
    
    print(f"\n🟢 LANGGRAPH AGENT SYSTEM:")
    print(f"   Sample Size: {len(agent_scores)}")
    print(f"   Mean Score: {agent_scores.mean():.3f}")
    print(f"   Std Dev: {agent_scores.std():.3f}")
    print(f"   Min Score: {agent_scores.min():.3f}")
    print(f"   Max Score: {agent_scores.max():.3f}")
    
    # Calculate improvement
    improvement = ((agent_scores.mean() - baseline_scores.mean()) / baseline_scores.mean()) * 100
    print(f"\n📈 MEAN IMPROVEMENT: {improvement:+.1f}%")
    
    # Perform Independent T-Test
    print(f"\n🔬 INDEPENDENT T-TEST:")
    t_stat, p_value = stats.ttest_ind(agent_scores, baseline_scores)
    print(f"   Test Statistic (t): {t_stat:.4f}")
    print(f"   P-Value (2-tailed): {p_value:.6f}")
    print(f"   Significance Level (α): 0.05")
    
    # Interpret result
    if p_value < 0.05:
        significance = "✅ STATISTICALLY SIGNIFICANT"
    else:
        significance = "❌ NOT STATISTICALLY SIGNIFICANT"
    
    print(f"   Result: {significance}")
    
    if p_value < 0.05:
        print(f"\n   ✅ We reject the null hypothesis.")
        print(f"   ✅ There is strong evidence that the agent system performs")
        print(f"      significantly better than the baseline (p < 0.05).")
    else:
        print(f"\n   ❌ We fail to reject the null hypothesis.")
        print(f"   ❌ There is insufficient evidence to claim a significant")
        print(f"      difference between the two systems.")
    
    # Calculate Confidence Interval
    print(f"\n📊 95% CONFIDENCE INTERVAL:")
    se_diff = np.sqrt(
        (baseline_scores.std()**2 / len(baseline_scores)) +
        (agent_scores.std()**2 / len(agent_scores))
    )
    ci_lower = (agent_scores.mean() - baseline_scores.mean()) - 1.96 * se_diff
    ci_upper = (agent_scores.mean() - baseline_scores.mean()) + 1.96 * se_diff
    print(f"   Difference in means: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Effect Size (Cohen's d)
    pooled_std = np.sqrt(
        ((len(baseline_scores)-1)*baseline_scores.std()**2 +
         (len(agent_scores)-1)*agent_scores.std()**2) /
        (len(baseline_scores) + len(agent_scores) - 2)
    )
    cohens_d = (agent_scores.mean() - baseline_scores.mean()) / pooled_std
    print(f"\n📏 EFFECT SIZE (Cohen's d): {cohens_d:.4f}")
    if abs(cohens_d) < 0.2:
        effect = "Small"
    elif abs(cohens_d) < 0.5:
        effect = "Small to Medium"
    elif abs(cohens_d) < 0.8:
        effect = "Medium to Large"
    else:
        effect = "Large"
    print(f"   Interpretation: {effect} effect")
    
    # Mann-Whitney U Test (non-parametric alternative)
    u_stat, u_pvalue = stats.mannwhitneyu(agent_scores, baseline_scores)
    print(f"\n🔬 MANN-WHITNEY U TEST (non-parametric):")
    print(f"   U Statistic: {u_stat:.4f}")
    print(f"   P-Value: {u_pvalue:.6f}")
    
    # Summary Report
        # Summary Report
    report = {
        'timestamp': datetime.now().isoformat(),
        'baseline': {
            'mean': float(baseline_scores.mean()),
            'std': float(baseline_scores.std()),
            'n': len(baseline_scores)
        },
        'agent': {
            'mean': float(agent_scores.mean()),
            'std': float(agent_scores.std()),
            'n': len(agent_scores)
        },
        'improvement_percent': float(improvement),
        't_test': {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_at_05': bool(p_value < 0.05)  # <--- FIXED HERE
        },
        'mannwhitney_u_test': {
            'u_statistic': float(u_stat),
            'p_value': float(u_pvalue),
            'significant_at_05': bool(u_pvalue < 0.05)  # <--- AND FIXED HERE
        },
        'cohens_d': float(cohens_d),
        'effect_size': effect,
        'confidence_interval_95': {
            'lower': float(ci_lower),
            'upper': float(ci_upper)
        }
    }

    
    # Save report
    with open('ab_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Full A/B test report saved: ab_test_report.json")
    
    return report

import os

if __name__ == "__main__":
    print("Starting A/B Statistical Analysis...")
    perform_ab_test()
