from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import json
import os
import subprocess
import sys
import torch
from summarize_results import write_outputs

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
PROFILES = {'baby': {'n_layers': 2, 'fusion_coeff': 0.4, 'cl_weight': 0.005, 'reg_weight': 0.0000003}, 'sports': {'n_layers': 3, 'fusion_coeff': 0.3, 'cl_weight': 0.005, 'reg_weight': 0.00000005}}

def command_for(args, base, variant, model, seed, result_file, log_file, checkpoint_dir):
    profile = PROFILES[base]
    command = [sys.executable, 'main.py', '--model', model, '--dataset', variant, '--seed', str(seed), '--gpu-id', str(args.gpu_id), '--epochs', str(args.epochs), '--stopping-step', str(args.stopping_step), '--train-batch-size', str(args.train_batch_size), '--eval-batch-size', str(args.eval_batch_size), '--embedding-size', str(args.embedding_size), '--n-layers', str(profile['n_layers']), '--fusion-coeff', str(profile['fusion_coeff']), '--cl-weight', str(profile['cl_weight']), '--reg-weight', str(profile['reg_weight']), '--knn-build-batch-size', str(args.knn_build_batch_size), '--reliability-temperature', str(args.reliability_temperature), '--reliability-shrinkage', str(args.reliability_shrinkage), '--checkpoint-dir', str(checkpoint_dir), '--log-file', str(log_file), '--result-file', str(result_file)]
    if args.cpu:
        command.append('--cpu')
    if args.save_model:
        command.append('--save-model')
    return command

def main():
    parser = ArgumentParser()
    parser.add_argument('--datasets', nargs='+', choices=sorted(PROFILES), default=['baby'])
    parser.add_argument('--noise-levels', nargs='+', type=int, default=[0, 10, 20, 30])
    parser.add_argument('--models', nargs='+', choices=['MSCA', 'RAMSCA'], default=['MSCA', 'RAMSCA'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[999])
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--stopping-step', type=int, default=20)
    parser.add_argument('--train-batch-size', type=int, default=2048)
    parser.add_argument('--eval-batch-size', type=int, default=1024)
    parser.add_argument('--embedding-size', type=int, default=64)
    parser.add_argument('--knn-build-batch-size', type=int, default=1024)
    parser.add_argument('--reliability-temperature', type=float, default=0.2)
    parser.add_argument('--reliability-shrinkage', type=float, default=0.5)
    parser.add_argument('--output-dir')
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--continue-on-error', action='store_true')
    args = parser.parse_args()
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available; install a CUDA build of PyTorch or use --cpu')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (ROOT / 'outputs' / stamp).resolve()
    run_dir = output_dir / 'runs'
    log_dir = output_dir / 'logs'
    checkpoint_dir = output_dir / 'checkpoints'
    for path in (run_dir, log_dir, checkpoint_dir):
        path.mkdir(parents=True, exist_ok=True)
    configuration = vars(args).copy()
    configuration['output_dir'] = str(output_dir)
    configuration['profiles'] = {name: PROFILES[name] for name in args.datasets}
    (output_dir / 'run_config.json').write_text(json.dumps(configuration, indent=2), encoding='utf-8')
    environment = os.environ.copy()
    failures = []
    for base in args.datasets:
        for noise in sorted(set(args.noise_levels)):
            variant = f'{base}_n{noise}'
            if not (ROOT / 'data' / variant).is_dir():
                raise FileNotFoundError(f'data/{variant} is missing; run prepare_data.py first')
            for seed in args.seeds:
                environment['PYTHONHASHSEED'] = str(seed)
                for model in args.models:
                    name = f'{model.lower()}__{variant}__seed{seed}'
                    result_file = run_dir / f'{name}.json'
                    log_file = log_dir / f'{name}.log'
                    command = command_for(args, base, variant, model, seed, result_file, log_file, checkpoint_dir)
                    print(' '.join(command), flush=True)
                    try:
                        subprocess.run(command, cwd=SRC, env=environment, check=True)
                        write_outputs(output_dir)
                    except subprocess.CalledProcessError as error:
                        failure = {'model': model, 'dataset': variant, 'seed': seed, 'returncode': error.returncode, 'command': command}
                        failures.append(failure)
                        (output_dir / 'failures.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
                        if not args.continue_on_error:
                            raise
    rows = write_outputs(output_dir)
    print(json.dumps({'completed_runs': len(rows), 'failures': len(failures), 'output_dir': str(output_dir)}))

if __name__ == '__main__':
    main()
