from argparse import ArgumentParser
from pathlib import Path
import json
import os
import time
import torch
from utils.quick_start import quick_start

os.environ['NUMEXPR_MAX_THREADS'] = '48'

def main():
    parser = ArgumentParser()
    parser.add_argument('--model', '-m', choices=['MSCA', 'RAMSCA'], required=True)
    parser.add_argument('--dataset', '-d', required=True)
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--stopping-step', type=int, default=20)
    parser.add_argument('--train-batch-size', type=int, default=2048)
    parser.add_argument('--eval-batch-size', type=int, default=1024)
    parser.add_argument('--embedding-size', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--fusion-coeff', type=float, default=0.4)
    parser.add_argument('--cl-weight', type=float, default=0.005)
    parser.add_argument('--reg-weight', type=float, default=0.0000003)
    parser.add_argument('--knn-build-batch-size', type=int, default=1024)
    parser.add_argument('--reliability-temperature', type=float, default=0.2)
    parser.add_argument('--reliability-shrinkage', type=float, default=0.5)
    parser.add_argument('--checkpoint-dir', default='../outputs/checkpoints')
    parser.add_argument('--log-file', default='./log')
    parser.add_argument('--result-file', required=True)
    parser.add_argument('--save-model', action='store_true')
    args = parser.parse_args()
    config = {'gpu_id': args.gpu_id, 'use_gpu': not args.cpu, 'seed': [args.seed], 'epochs': args.epochs, 'stopping_step': args.stopping_step, 'train_batch_size': args.train_batch_size, 'eval_batch_size': args.eval_batch_size, 'embedding_size': args.embedding_size, 'n_layers': [args.n_layers], 'fusion_coeff': [args.fusion_coeff], 'cl_weight': [args.cl_weight], 'reg_weight': [args.reg_weight], 'knn_build_batch_size': args.knn_build_batch_size, 'reliability_temperature': args.reliability_temperature, 'reliability_shrinkage': args.reliability_shrinkage, 'checkpoint_dir': args.checkpoint_dir, 'log_dir': args.log_file}
    started = time.time()
    result = quick_start(args.model, args.dataset, config, save_model=args.save_model)
    result['duration_seconds'] = time.time() - started
    result['torch_version'] = torch.__version__
    result['cuda_version'] = torch.version.cuda
    result['gpu_name'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() and not args.cpu else None
    target = Path(args.result_file).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, default=lambda value: value.item() if hasattr(value, 'item') else str(value)), encoding='utf-8')
    print(json.dumps({'result_file': str(target), 'duration_seconds': result['duration_seconds']}))

if __name__ == '__main__':
    main()
