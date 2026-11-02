import os
import sys
import subprocess
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="D-FINE Fine-Tuning Runner on UAVSwarm")
    parser.add_argument('--config', default='/content/D-FINE/configs/dfine/custom/dfine_hgnetv2_m_uavswarm.yml')
    parser.add_argument('--weights', default='/content/weights/dfine_m_coco.pth')
    parser.add_argument('--epochs', type=int, default=24)
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("🎯 Launching D-FINE UAVSwarm Fine-Tuning", flush=True)
    print(f"Config: {args.config}", flush=True)
    print(f"Weights: {args.weights}", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)
    print("=" * 60, flush=True)

    cmd = [
        sys.executable, '/content/D-FINE/train.py',
        '-c', args.config,
        '-t', args.weights,
        '--use-amp',
        '-u', f'epochs={args.epochs}'
    ]

    env = os.environ.copy()
    env['PYTHONPATH'] = '/content/D-FINE:' + env.get('PYTHONPATH', '')

    t0 = time.time()
    res = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0

    if res.returncode == 0:
        print(f"\n🎉 D-FINE Fine-Tuning Completed Successfully in {elapsed / 3600:.2f} hours!", flush=True)
    else:
        print(f"\n❌ D-FINE Training Exited with return code {res.returncode}", flush=True)
        sys.exit(res.returncode)

if __name__ == '__main__':
    main()
