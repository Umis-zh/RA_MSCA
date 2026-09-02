from argparse import ArgumentParser
from pathlib import Path
import json
import os
import shutil
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent

def replace_file(path, overwrite):
    if path.exists():
        if not overwrite:
            raise FileExistsError(f'{path} already exists; use --overwrite to rebuild')
        path.unlink()

def link_or_copy(source, target, overwrite):
    replace_file(target, overwrite)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)

def write_variant_config(dataset, variant):
    source = ROOT / 'src' / 'configs' / 'dataset' / f'{dataset}.yaml'
    target = ROOT / 'src' / 'configs' / 'dataset' / f'{variant}.yaml'
    config = yaml.safe_load(source.read_text(encoding='utf-8'))
    config['inter_file_name'] = f'{variant}.inter'
    target.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')

def prepare_dataset(dataset, data_root, levels, seed, overwrite, chunk_size):
    source = data_root / dataset
    inter_path = source / f'{dataset}.inter'
    image_path = source / 'image_feat.npy'
    text_path = source / 'text_feat.npy'
    for path in (inter_path, image_path, text_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    frame = pd.read_csv(inter_path, sep='\t')
    if not {'userID', 'itemID', 'x_label'}.issubset(frame.columns):
        raise ValueError(f'{inter_path} must contain userID, itemID and x_label')
    image = np.load(image_path, mmap_mode='r')
    text = np.load(text_path, mmap_mode='r')
    if image.shape[0] != text.shape[0]:
        raise ValueError(f'feature row mismatch in {dataset}')
    if int(frame['itemID'].max()) >= image.shape[0]:
        raise ValueError(f'item index exceeds feature rows in {dataset}')
    generator = np.random.default_rng(seed)
    corruption_order = generator.permutation(image.shape[0])
    donor_order = generator.permutation(image.shape[0])
    while np.any(corruption_order == donor_order):
        donor_order = generator.permutation(image.shape[0])
    map_path = data_root / f'{dataset}_noise_map.npz'
    replace_file(map_path, overwrite)
    np.savez_compressed(map_path, corruption_order=corruption_order, donor_order=donor_order, seed=np.array([seed]))
    variants = []
    for level in levels:
        variant = f'{dataset}_n{level}'
        target = data_root / variant
        target.mkdir(parents=True, exist_ok=True)
        target_inter = target / f'{variant}.inter'
        target_image = target / 'image_feat.npy'
        target_text = target / 'text_feat.npy'
        link_or_copy(inter_path, target_inter, overwrite)
        link_or_copy(text_path, target_text, overwrite)
        if level == 0:
            link_or_copy(image_path, target_image, overwrite)
        else:
            replace_file(target_image, overwrite)
            shutil.copy2(image_path, target_image)
            noisy = np.load(target_image, mmap_mode='r+')
            count = int(round(image.shape[0] * level / 100))
            for start in range(0, count, chunk_size):
                end = min(start + chunk_size, count)
                targets = corruption_order[start:end]
                donors = donor_order[start:end]
                noisy[targets] = image[donors]
            noisy.flush()
            del noisy
        for cache in target.glob('*_adj_*.pt'):
            cache.unlink()
        write_variant_config(dataset, variant)
        count = int(round(image.shape[0] * level / 100))
        variants.append({'dataset': dataset, 'variant': variant, 'noise_percent': level, 'users': int(frame['userID'].nunique()), 'items': int(image.shape[0]), 'interactions': int(len(frame)), 'corrupted_items': count, 'image_shape': list(image.shape), 'text_shape': list(text.shape)})
    return variants

def main():
    parser = ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['baby'])
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--noise-levels', nargs='+', type=int, default=[0, 10, 20, 30])
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--chunk-size', type=int, default=1024)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if any((level < 0 or level > 100 for level in args.noise_levels)):
        raise ValueError('noise levels must be between 0 and 100')
    data_root = (ROOT / args.data_root).resolve()
    records = []
    for dataset in args.datasets:
        records.extend(prepare_dataset(dataset, data_root, sorted(set(args.noise_levels)), args.seed, args.overwrite, args.chunk_size))
    manifest = {'seed': args.seed, 'nested_corruption': True, 'records': records}
    target = data_root / 'noise_manifest.json'
    target.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
