"""
Run EQL on all three Lorenz equations sequentially
Automatically generates summaries and visualizations after completion
"""
import subprocess
import sys
import time
import os
import shutil

# Commands to run
commands = [
    {
        'name': 'Lorenz dx/dt equation',
        'cmd': [
            sys.executable,
            'experiment/evaluate_model.py',
            'data/lorenz/lorenz_dx.tsv.gz',
            '-ml', 'eql',
            '-results_path', 'results/lorenz',
            '-seed', '42'
        ],
        'output_name': 'lorenz_dx_eql'
    },
    {
        'name': 'Lorenz dy/dt equation',
        'cmd': [
            sys.executable,
            'experiment/evaluate_model.py',
            'data/lorenz/lorenz_dy.tsv.gz',
            '-ml', 'eql',
            '-results_path', 'results/lorenz',
            '-seed', '42'
        ],
        'output_name': 'lorenz_dy_eql'
    },
    {
        'name': 'Lorenz dz/dt equation',
        'cmd': [
            sys.executable,
            'experiment/evaluate_model.py',
            'data/lorenz/lorenz_dz.tsv.gz',
            '-ml', 'eql',
            '-results_path', 'results/lorenz',
            '-seed', '42'
        ],
        'output_name': 'lorenz_dz_eql'
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

def organize_results():
    """Organize results from results/lorenz to outputs/lorenz_*_eql"""
    print("\n" + "="*80)
    print("Organizing results into outputs/ directory...")
    print("="*80)

    for cmd_info in commands:
        eq = cmd_info['output_name'].split('_')[1]  # Extract dx, dy, or dz
        source_json = f'results/lorenz/lorenz_{eq}_eql_42.json'
        target_dir = f'outputs/{cmd_info["output_name"]}'

        if os.path.exists(source_json):
            # Create target directory
            os.makedirs(target_dir, exist_ok=True)

            # Copy the JSON result file
            target_json = os.path.join(target_dir, f'eql_result.json')
            shutil.copy2(source_json, target_json)

            print(f"  ✅ Organized results for {eq}")
            print(f"     Source: {source_json}")
            print(f"     Target: {target_json}")
        else:
            print(f"  ⚠️  Warning: {source_json} not found")

def generate_summary():
    """Run summary script"""
    print("\n" + "="*80)
    print("Generating summary...")
    print("="*80)

    try:
        result = subprocess.run(
            [sys.executable, 'summarize_eql_results.py'],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)

        # Save to file
        with open('outputs/eql_results_summary.txt', 'w', encoding='utf-8') as f:
            f.write(result.stdout)

        print("✅ Summary generated successfully")
        print("   Saved to: outputs/eql_results_summary.txt")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Summary generation failed: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")

def generate_visualization():
    """Run visualization script"""
    print("\n" + "="*80)
    print("Generating visualization...")
    print("="*80)

    try:
        subprocess.run([sys.executable, 'visualize_lorenz_eql.py'], check=True)
        print("✅ Visualization generated successfully")
        print("   Saved to: outputs/lorenz_comparison_eql.png")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Visualization generation failed: {e}")

if __name__ == "__main__":
    print("="*80)
    print("EQL Lorenz System Discovery Experiment")
    print("="*80)
    print(f"Total equations to process: {len(commands)}")
    print(f"Using full function set: sin, cos, exp, log, sqrt, div, mul")
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
        organize_results()

        # Generate summary and visualization
        generate_summary()
        generate_visualization()

        print("\n" + "="*80)
        print("🎉 All tasks completed!")
        print("="*80)
        print("\nGenerated files:")
        print("  - results/lorenz/lorenz_dx_eql_42.json")
        print("  - results/lorenz/lorenz_dy_eql_42.json")
        print("  - results/lorenz/lorenz_dz_eql_42.json")
        print("  - outputs/lorenz_dx_eql/eql_result.json")
        print("  - outputs/lorenz_dy_eql/eql_result.json")
        print("  - outputs/lorenz_dz_eql/eql_result.json")
        print("  - outputs/eql_results_summary.txt")
        print("  - outputs/lorenz_comparison_eql.png")
        print("="*80)
    else:
        print("\n⚠️  Some experiments failed. Please check the errors above.")
        sys.exit(1)
