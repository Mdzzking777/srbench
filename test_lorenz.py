"""
测试 Lorenz 数据集的脚本
直接在 VSCode 中按 F5 或右键"运行"即可！
"""

import subprocess
import sys

print("=" * 60)
print("开始测试 Lorenz 数据集")
print("=" * 60)

# 测试命令
commands = [
    {
        'name': '快速测试 lorenz_dx (PySR)',
        'cmd': [
            sys.executable,  # Python 解释器路径
            'experiment/evaluate_model.py',
            'data/lorenz/lorenz_dx.tsv.gz',
            '-ml', 'pysr',
            '-results_path', 'results/lorenz',
            '-seed', '42',
            '-test'  # 快速测试模式
        ]
    }
]

# 运行测试
for test in commands:
    print(f"\n{'='*60}")
    print(f"运行: {test['name']}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            test['cmd'],
            cwd='.',  # 当前目录
            capture_output=False,  # 直接显示输出
            text=True
        )

        if result.returncode == 0:
            print(f"\n✅ {test['name']} 完成！")
        else:
            print(f"\n❌ {test['name']} 失败 (错误代码: {result.returncode})")

    except Exception as e:
        print(f"\n❌ 运行错误: {e}")

print("\n" + "=" * 60)
print("测试完成！")
print("结果保存在: results/lorenz/")
print("=" * 60)
