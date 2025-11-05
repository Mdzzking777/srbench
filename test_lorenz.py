"""
测试 Lorenz 数据集的脚本
直接在 VSCode 中按 F5 或右键"运行"即可！
"""

import subprocess
import sys
import os
import glob
import shutil

# 设置Julia环境到用户目录（避免权限问题）
os.environ['JULIA_DEPOT_PATH'] = os.path.expanduser('~/.julia')
os.environ['PYTHON_JULIACALL_DEPOT'] = os.path.expanduser('~/.julia')

print("=" * 60)
print("开始测试 Lorenz 数据集")
print("=" * 60)

# 测试命令
commands = [
    {
        'name': '完整运行 lorenz_dx (PySR) - 预期: σ(y-x)',
        'output_name': 'lorenz_dx',
        'cmd': [
            sys.executable,
            'experiment/evaluate_model.py',
            'data/lorenz/lorenz_dx.tsv.gz',
            '-ml', 'pysr',
            '-results_path', 'results/lorenz',
            '-seed', '42',
            # '-test'  # 快速测试：3次迭代
        ]
    },
    {
        'name': '完整运行 lorenz_dy (PySR) - 预期: x(ρ-z) - y',
        'output_name': 'lorenz_dy',
        'cmd': [
            sys.executable,
            'experiment/evaluate_model.py',
            'data/lorenz/lorenz_dy.tsv.gz',
            '-ml', 'pysr',
            '-results_path', 'results/lorenz',
            '-seed', '42',
            # '-test'
        ]
    },
    {
        'name': '完整运行 lorenz_dz (PySR) - 预期: xy - βz',
        'output_name': 'lorenz_dz',
        'cmd': [
            sys.executable,
            'experiment/evaluate_model.py',
            'data/lorenz/lorenz_dz.tsv.gz',
            '-ml', 'pysr',
            '-results_path', 'results/lorenz',
            '-seed', '42',
            # '-test'
        ]
    }
]

# 运行测试
for test in commands:
    print(f"\n{'='*60}")
    print(f"运行: {test['name']}")
    print(f"{'='*60}\n")

    # 记录运行前的 outputs 文件夹内容
    outputs_before = set(os.listdir('outputs')) if os.path.exists('outputs') else set()

    try:
        # 创建环境变量副本，确保Julia使用用户目录
        env = os.environ.copy()
        env['JULIA_DEPOT_PATH'] = os.path.expanduser('~/.julia')
        env['PYTHON_JULIACALL_DEPOT'] = os.path.expanduser('~/.julia')

        result = subprocess.run(
            test['cmd'],
            cwd='.',  # 当前目录
            capture_output=False,  # 直接显示输出
            text=True,
            env=env  # 显式传递环境变量
        )

        if result.returncode == 0:
            print(f"\n✅ {test['name']} 完成！")

            # 重命名新生成的 outputs 文件夹
            outputs_after = set(os.listdir('outputs')) if os.path.exists('outputs') else set()
            new_folders = outputs_after - outputs_before

            if new_folders:
                # 找到最新的文件夹（通常只有一个）
                latest_folder = sorted(new_folders)[-1]
                old_path = os.path.join('outputs', latest_folder)
                new_path = os.path.join('outputs', test['output_name'])

                # 如果目标文件夹已存在，先删除
                if os.path.exists(new_path):
                    shutil.rmtree(new_path)

                # 重命名
                shutil.move(old_path, new_path)
                print(f"📁 输出文件夹已重命名: outputs/{test['output_name']}/")
        else:
            print(f"\n❌ {test['name']} 失败 (错误代码: {result.returncode})")

            # 即使失败，也尝试重命名输出文件夹（可能有部分结果）
            outputs_after = set(os.listdir('outputs')) if os.path.exists('outputs') else set()
            new_folders = outputs_after - outputs_before

            if new_folders:
                latest_folder = sorted(new_folders)[-1]
                old_path = os.path.join('outputs', latest_folder)
                new_path = os.path.join('outputs', test['output_name'] + '_failed')

                if os.path.exists(new_path):
                    shutil.rmtree(new_path)

                shutil.move(old_path, new_path)
                print(f"📁 失败的输出文件夹已保存: outputs/{test['output_name']}_failed/")

    except Exception as e:
        print(f"\n❌ 运行错误: {e}")

print("\n" + "=" * 60)
print("测试完成！")
print("结果保存在: results/lorenz/")
print("Hall of Fame 保存在: outputs/lorenz_dx/, outputs/lorenz_dy/, outputs/lorenz_dz/")
print("=" * 60)
