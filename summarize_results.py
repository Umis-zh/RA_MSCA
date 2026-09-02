from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import csv
import json
import statistics

METRICS = ('recall@10', 'recall@20', 'ndcg@10', 'ndcg@20', 'precision@20', 'map@20')

def load_rows(run_dir):
    rows = []
    run_dir = Path(run_dir)
    output_dir = run_dir.parent
    for path in sorted(run_dir.glob('*.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        best = payload['runs'][payload['best_run_index']]
        dataset = payload['dataset']
        base, noise = dataset.rsplit('_n', 1)
        row = {'base_dataset': base, 'dataset': dataset, 'noise_percent': int(noise), 'model': payload['model'], 'seed': int(best['parameters']['seed']), 'best_epoch': int(best['best_epoch']), 'device': payload['device'], 'gpu_name': payload.get('gpu_name') or '', 'duration_seconds': round(float(payload['duration_seconds']), 3), 'result_file': str(path.relative_to(output_dir)), 'log_file': str(Path('logs') / Path(payload['log_file']).name)}
        for metric in METRICS:
            row[metric] = float(best['test'][metric])
        rows.append(row)
    return rows

def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['base_dataset'], row['noise_percent'], row['model'])].append(row)
    output = []
    for key in sorted(grouped):
        group = grouped[key]
        item = {'base_dataset': key[0], 'noise_percent': key[1], 'model': key[2], 'runs': len(group)}
        for metric in METRICS:
            values = [row[metric] for row in group]
            item[f'{metric}_mean'] = round(statistics.fmean(values), 8)
            item[f'{metric}_std'] = round(statistics.stdev(values), 8) if len(values) > 1 else 0.0
        output.append(item)
    return output

def compare(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row['base_dataset'], row['noise_percent'], row['seed'])][row['model']] = row
    output = []
    for key in sorted(grouped):
        pair = grouped[key]
        if 'MSCA' not in pair or 'RAMSCA' not in pair:
            continue
        item = {'base_dataset': key[0], 'noise_percent': key[1], 'seed': key[2]}
        for metric in METRICS:
            baseline = pair['MSCA'][metric]
            improved = pair['RAMSCA'][metric]
            item[f'msca_{metric}'] = baseline
            item[f'ramsca_{metric}'] = improved
            item[f'delta_{metric}'] = round(improved - baseline, 8)
            item[f'relative_{metric}_percent'] = round((improved - baseline) / baseline * 100, 6) if baseline else 0.0
        output.append(item)
    return output

def write_outputs(output_dir):
    output_dir = Path(output_dir)
    rows = load_rows(output_dir / 'runs')
    if not rows:
        return []
    write_csv(output_dir / 'runs.csv', rows, list(rows[0].keys()))
    summary = aggregate(rows)
    write_csv(output_dir / 'summary.csv', summary, list(summary[0].keys()))
    comparison = compare(rows)
    if comparison:
        write_csv(output_dir / 'comparison.csv', comparison, list(comparison[0].keys()))
    return rows

def main():
    parser = ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    rows = write_outputs(args.output_dir)
    print(json.dumps({'runs': len(rows), 'output_dir': str(Path(args.output_dir).resolve())}))

if __name__ == '__main__':
    main()
