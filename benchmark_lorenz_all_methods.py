"""
使用SRBENCH所有符号回归方法测试Lorenz数据集
完整基准测试 - 比较所有方法的性能
"""

import subprocess
import sys
import os
import json
from pathlib import Path

# 设置Julia环境到用户目录（避免权限问题）
os.environ['JULIA_DEPOT_PATH'] = os.path.expanduser('~/.julia')
os.environ['PYTHON_JULIACALL_DEPOT'] = os.path.expanduser('~/.julia')

print("=" * 80)
print("SRBENCH Lorenz 完整基准测试")
print("=" * 80)

# SRBENCH中所有符号回归方法
# 来源: README.md和algorithms/目录
SR_METHODS = [
    'pysr',           # PySR (推荐)
    'feat',           # Feature Engineering Automation Tool
    'operon',         # Operon
    'gplearn',        # gplearn
    'afp',            # Age-Fitness Pareto Optimization
    'afp-FE',         # AFP with Fitness Predictors
    'eplex',          # epsilon-Lexicase Selection
    'FFX',            # Fast Function Extraction
    'BSR',            # Bayesian Symbolic Regression
    'DSR',            # Deep Symbolic Regression
    'AIFeynman',      # AI Feynman 2.0
    'ITEA',           # Interaction-Transformation EA
    'MRGP',           # Multiple Regression GP
    'GP-GOMEA',       # GP-based GOMEA
    'SBP-GOMEA',      # Semantic Backprop GP
]

# Lorenz数据集
DATASETS = [
    ('data/lorenz/lorenz_dx.tsv.gz', 'dx/dt = σ(y-x)'),
    ('data/lorenz/lorenz_dy.tsv.gz', 'dy/dt = x(ρ-z) - y'),
    ('data/lorenz/lorenz_dz.tsv.gz', 'dz/dt = xy - βz'),
]

# 测试模式选择
USE_TEST_MODE = True  # 改为False进行完整运行（每个方法2小时）
SEED = 42
RESULTS_DIR = 'results/lorenz_benchmark'

# 重要提示：
# - 只有PySR定义了test_params（3次迭代，~2分钟）
# - 其他14种方法没有test_params，-test标志可能无效
# - 在测试模式下，我们对非PySR方法强制10分钟超时
#   以避免等待过长（虽然可能中断运行）

print(f"测试模式: {'快速测试 (建议先用)' if USE_TEST_MODE else '完整运行 (耗时长)'}")
print(f"方法数量: {len(SR_METHODS)}")
print(f"数据集: {len(DATASETS)}")
print(f"总任务数: {len(SR_METHODS) * len(DATASETS)}")
if USE_TEST_MODE:
    print(f"预计时间: ~{len(SR_METHODS) * len(DATASETS) * 2} 分钟")
else:
    print(f"预计时间: ~{len(SR_METHODS) * len(DATASETS) * 2} 小时")
print("=" * 80)

# 创建结果目录
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# 统计结果
results_summary = []
failed_runs = []

# 遍历所有方法和数据集
for method in SR_METHODS:
    for dataset, equation in DATASETS:
        dataset_name = Path(dataset).stem

        print(f"\n{'='*80}")
        print(f"方法: {method:15s} | 数据集: {dataset_name:15s} | 方程: {equation}")
        print(f"{'='*80}")

        # 构建命令
        cmd = [
            sys.executable,
            'experiment/evaluate_model.py',
            dataset,
            '-ml', method,
            '-results_path', RESULTS_DIR,
            '-seed', str(SEED),
        ]

        if USE_TEST_MODE:
            cmd.append('-test')

        # 创建环境变量副本
        env = os.environ.copy()
        env['JULIA_DEPOT_PATH'] = os.path.expanduser('~/.julia')
        env['PYTHON_JULIACALL_DEPOT'] = os.path.expanduser('~/.julia')

        try:
            # 设置超时：PySR的-test很快，但其他方法可能没有test_params
            # 在测试模式下强制超时以避免等太久
            if USE_TEST_MODE:
                timeout = 300 if method == 'pysr' else 600  # PySR: 5分钟，其他: 10分钟
            else:
                timeout = 7200  # 完整运行: 2小时

            print(f"超时设置: {timeout//60}分钟")

            result = subprocess.run(
                cmd,
                cwd='.',
                capture_output=False,
                text=True,
                env=env,
                timeout=timeout
            )

            if result.returncode == 0:
                print(f"\n✅ {method} on {dataset_name} 完成")

                # 尝试读取结果
                result_file = Path(RESULTS_DIR) / f"{dataset_name}_{method}_{SEED}.json"
                if result_file.exists():
                    with open(result_file) as f:
                        result_data = json.load(f)
                        results_summary.append({
                            'method': method,
                            'dataset': dataset_name,
                            'equation': equation,
                            'model': result_data.get('symbolic_model', 'N/A'),
                            'r2_test': result_data.get('r2_test', -999),
                            'simplicity': result_data.get('simplicity', -999),
                        })
            else:
                print(f"\n❌ {method} on {dataset_name} 失败 (错误码: {result.returncode})")
                failed_runs.append((method, dataset_name))

        except subprocess.TimeoutExpired:
            print(f"\n⏰ {method} on {dataset_name} 超时")
            failed_runs.append((method, dataset_name))
        except Exception as e:
            print(f"\n❌ {method} on {dataset_name} 错误: {e}")
            failed_runs.append((method, dataset_name))

# 输出总结
print("\n" + "=" * 80)
print("基准测试完成！")
print("=" * 80)

print(f"\n成功: {len(results_summary)}/{len(SR_METHODS) * len(DATASETS)}")
print(f"失败: {len(failed_runs)}")

if failed_runs:
    print("\n失败的运行:")
    for method, dataset in failed_runs:
        print(f"  - {method} on {dataset}")

# 按R²排序结果
if results_summary:
    print("\n" + "=" * 80)
    print("排名 (按测试集R²降序)")
    print("=" * 80)

    for dataset_filter in ['lorenz_dx', 'lorenz_dy', 'lorenz_dz']:
        dataset_results = [r for r in results_summary if r['dataset'] == dataset_filter]
        if not dataset_results:
            continue

        dataset_results.sort(key=lambda x: x['r2_test'], reverse=True)

        print(f"\n{dataset_filter} (目标: {[r['equation'] for r in dataset_results][0]})")
        print("-" * 80)
        print(f"{'排名':<5} {'方法':<15} {'R²测试':<10} {'简洁度':<10} {'模型'}")
        print("-" * 80)

        for rank, res in enumerate(dataset_results, 1):
            model_str = res['model'][:50] + '...' if len(res['model']) > 50 else res['model']
            print(f"{rank:<5} {res['method']:<15} {res['r2_test']:<10.4f} "
                  f"{res['simplicity']:<10.2f} {model_str}")

# 保存详细结果
summary_file = Path(RESULTS_DIR) / 'benchmark_summary.json'
with open(summary_file, 'w') as f:
    json.dump({
        'results': results_summary,
        'failed': failed_runs,
        'config': {
            'test_mode': USE_TEST_MODE,
            'seed': SEED,
            'methods': SR_METHODS,
            'datasets': [d[0] for d in DATASETS],
        }
    }, f, indent=2)

print(f"\n详细结果已保存: {summary_file}")
print(f"所有结果文件在: {RESULTS_DIR}/")
