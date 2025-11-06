"""
分析SRBENCH中各算法的表现
"""
import pandas as pd
import numpy as np

# 读取结果
print("="*60)
print("分析SRBENCH算法表现")
print("="*60)

try:
    # 读取ground-truth结果（有真实公式的数据集）
    df_gt = pd.read_feather('results/ground-truth_results.feather')
    print(f"\n✅ Ground-truth数据集结果: {len(df_gt)} 条记录")

    # 按算法分组，计算平均R²
    if 'algorithm' in df_gt.columns and 'r2_test' in df_gt.columns:
        method_performance = df_gt.groupby('algorithm')['r2_test'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False)

        print("\n" + "="*60)
        print("算法排名（按平均R²测试分数）")
        print("="*60)
        print(f"{'排名':<5} {'算法':<25} {'平均R²':<12} {'中位R²':<12} {'数据集数'}")
        print("-"*60)

        for i, (method, row) in enumerate(method_performance.head(15).iterrows(), 1):
            print(f"{i:<5} {method:<25} {row['mean']:<12.4f} {row['median']:<12.4f} {int(row['count'])}")

    # 读取black-box结果（无真实公式的数据集）
    df_bb = pd.read_feather('results/black-box_results.feather')
    print(f"\n✅ Black-box数据集结果: {len(df_bb)} 条记录")

    if 'algorithm' in df_bb.columns and 'r2_test' in df_bb.columns:
        method_performance_bb = df_bb.groupby('algorithm')['r2_test'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False)

        print("\n" + "="*60)
        print("Black-box数据集排名（按平均R²）")
        print("="*60)
        print(f"{'排名':<5} {'算法':<25} {'平均R²':<12} {'中位R²':<12} {'数据集数'}")
        print("-"*60)

        for i, (method, row) in enumerate(method_performance_bb.head(15).iterrows(), 1):
            print(f"{i:<5} {method:<25} {row['mean']:<12.4f} {row['median']:<12.4f} {int(row['count'])}")

except Exception as e:
    print(f"\n❌ 错误: {e}")
    print("\n可用的列:")
    print(df_gt.columns.tolist() if 'df_gt' in locals() else "无法读取数据")

print("\n" + "="*60)
