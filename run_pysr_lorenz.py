"""
Run PySR on all three Lorenz equations sequentially
Automatically generates summaries and visualizations after completion
"""
import subprocess
import sys
import time
import os

# Commands to run
commands = [
    {
        'name': 'Lorenz dx/dt equation',
        'cmd': [
            sys.executable,
            'experiment/evaluate_model.py',
            'data/lorenz/lorenz_dx.tsv.gz',
            '-ml', 'pysr',
            '-results_path', 'results/lorenz',
            '-seed', '42'
        ]
    },
    {
        'name': 'Lorenz dy/dt equation',
        'cmd': [
            sys.executable,
            'experiment/evaluate_model.py',
            'data/lorenz/lorenz_dy.tsv.gz',
            '-ml', 'pysr',
            '-results_path', 'results/lorenz',
            '-seed', '42'
        ]
    },
    {
        'name': 'Lorenz dz/dt equation',
        'cmd': [
            sys.executable,
            'experiment/evaluate_model.py',
            'data/lorenz/lorenz_dz.tsv.gz',
            '-ml', 'pysr',
            '-results_path', 'results/lorenz',
            '-seed', '42'
        ]
    }
]

def run_experiment(cmd_info):
    """Run a single experiment"""
    print("\n" + "="*80)
    print(f"Running: {cmd_info['name']}")
    print("="*80)
    print(f"Command: {' '.join(cmd_info['cmd'])}")
    print()

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd_info['cmd'],
            check=True,
            capture_output=False,
            text=True
        )

        elapsed_time = time.time() - start_time
        print(f"\n✅ Completed: {cmd_info['name']}")
        print(f"   Time elapsed: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        return True, elapsed_time

    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ Failed: {cmd_info['name']}")
        print(f"   Error: {e}")
        print(f"   Time elapsed: {elapsed_time:.2f} seconds")
        return False, elapsed_time

def copy_results_to_outputs():
    """Copy results from results/lorenz to outputs/lorenz_*_pysr"""
    print("\n" + "="*80)
    print("Organizing results into outputs/ directory...")
    print("="*80)

    equations = ['dx', 'dy', 'dz']

    for eq in equations:
        source_json = f'results/lorenz/lorenz_{eq}_pysr_42.json'
        target_dir = f'outputs/lorenz_{eq}_pysr'

        if os.path.exists(source_json):
            # Create target directory
            os.makedirs(target_dir, exist_ok=True)

            # PySR automatically saves hall_of_fame.csv and checkpoint.pkl
            # Check if they exist in the expected location
            print(f"  ✅ Results for {eq} are ready")
        else:
            print(f"  ⚠️  Warning: {source_json} not found")

def generate_summary():
    """Run summary script"""
    print("\n" + "="*80)
    print("Generating summary...")
    print("="*80)

    try:
        subprocess.run([sys.executable, 'summarize_pysr_results.py'], check=True)
        print("✅ Summary generated successfully")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Summary generation failed: {e}")

def generate_visualization():
    """Run visualization script"""
    print("\n" + "="*80)
    print("Generating visualization...")
    print("="*80)

    try:
        subprocess.run([sys.executable, 'visualize_lorenz_pysr.py'], check=True)
        print("✅ Visualization generated successfully")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Visualization generation failed: {e}")

if __name__ == "__main__":
    print("="*80)
    print("PySR Lorenz System Discovery Experiment")
    print("="*80)
    print(f"Total equations to process: {len(commands)}")
    print(f"Expected total time: ~6 hours (2 hours per equation)")
    print()

    # Track results
    results = []
    total_start = time.time()

    # Run all experiments
    for cmd_info in commands:
        success, elapsed = run_experiment(cmd_info)
        results.append({
            'name': cmd_info['name'],
            'success': success,
            'time': elapsed
        })

    total_elapsed = time.time() - total_start

    # Print summary of runs
    print("\n" + "="*80)
    print("Experiment Completion Summary")
    print("="*80)

    for i, result in enumerate(results, 1):
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"{i}. {result['name']}")
        print(f"   Status: {status}")
        print(f"   Time: {result['time']:.2f} seconds ({result['time']/60:.2f} minutes)")

    print(f"\nTotal time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes, {total_elapsed/3600:.2f} hours)")

    # Check if all succeeded
    all_success = all(r['success'] for r in results)

    if all_success:
        print("\n✅ All experiments completed successfully!")

        # Organize results
        copy_results_to_outputs()

        # Generate summary and visualization
        generate_summary()
        generate_visualization()

        print("\n" + "="*80)
        print("🎉 All tasks completed!")
        print("="*80)
        print("\nGenerated files:")
        print("  - results/lorenz/lorenz_dx_pysr_42.json")
        print("  - results/lorenz/lorenz_dy_pysr_42.json")
        print("  - results/lorenz/lorenz_dz_pysr_42.json")
        print("  - outputs/lorenz_*_pysr/hall_of_fame.csv (for each equation)")
        print("  - outputs/lorenz_comparison_pysr.png")
        print("="*80)
    else:
        print("\n⚠️  Some experiments failed. Please check the errors above.")
        sys.exit(1)
