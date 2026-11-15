"""Package reviewed local 468-point targets; never infer or download labels.

Run from repository root: python -m scripts.import_registered_geometry --help
The staging index is JSONL with image, geometry, group, split. Source NPZs
must ALREADY satisfy centered-crop-width-xyz-468-v1. See experiment notes.
"""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

from veyra3d import GEOMETRY_CONTRACT
from veyra3d.config import DataConfig, ExperimentConfig
from veyra3d.data import local_path, make_datasets


def import_dataset(source, index, review, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Output must not exist; imports never overwrite data')
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Source and output must be separate, non-nested directories')
    approval = json.loads(local_path(source, review).read_text())
    if approval.get('contract') != GEOMETRY_CONTRACT:
        raise ValueError('Review must name the exact geometry contract')
    for key in ('license_reference', 'registration_reference', 'coordinate_conversion_reference'):
        if not isinstance(approval.get(key), str) or not approval[key].strip():
            raise ValueError(f'Review requires {key}')
    if approval.get('label_source') != 'scan-derived' or approval.get('reviewed') is not True:
        raise ValueError('Only explicitly reviewed scan-derived labels are accepted')
    records = [json.loads(line) for line in local_path(source, index).read_text().splitlines() if line.strip()]
    # Resolve and validate source paths before writing anything.
    prepared = []
    for record in records:
        if set(record) != {'image', 'geometry', 'group', 'split'} or record['split'] not in ('train', 'val', 'test'):
            raise ValueError('Index requires image, geometry, group, split=train|val|test')
        prepared.append((record, local_path(source, record['image']), local_path(source, record['geometry'])))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.geometry-import-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'dataset'
        (stage / 'crops').mkdir(parents=True)
        (stage / 'labels').mkdir()
        splits = {name: [] for name in ('train', 'val', 'test')}
        for number, (record, image, geometry) in enumerate(prepared):
            target_image = f'crops/{number:06d}{image.suffix.lower()}'
            target_geometry = f'labels/{number:06d}.npz'
            shutil.copyfile(image, stage / target_image)
            shutil.copyfile(geometry, stage / target_geometry)
            splits[record['split']].append({'image':target_image, 'geometry':target_geometry, 'group':record['group']})
        for name, rows in splits.items():
            (stage / f'{name}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        config = ExperimentConfig(data=DataConfig(kind='manifest', root=str(stage), test_manifest='test.jsonl'))
        datasets = make_datasets(config)  # Includes person and duplicate-image split checks.
        for dataset in datasets.values():
            for number in range(len(dataset)):
                dataset[number]  # Decode every image and validate every label; no model runs.
        (stage / 'import-review.json').write_text(json.dumps(approval, indent=2)+'\n')
        stage.rename(output)
    return {name:len(rows) for name, rows in splits.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--index', default='index.jsonl')
    parser.add_argument('--review', default='review.json')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(import_dataset(args.source, args.index, args.review, args.output)))


if __name__ == '__main__':
    main()
