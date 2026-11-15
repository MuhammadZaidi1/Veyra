"""Bounded synthetic-only CUDA training benchmark; no datasets or checkpoints."""
import json
import statistics
import time

import torch
from veyra3d.config import ExperimentConfig
from veyra3d.data import SyntheticDataset
from veyra3d.model import Veyra3D
from veyra3d.objectives import GeometryObjective
from veyra3d.runtime import environment, initialize


def main():
    config = ExperimentConfig(device='cuda', batch_size=8)
    device = initialize(config)
    model = Veyra3D(config.model).to(device)
    objective = GeometryObjective().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    samples = [SyntheticDataset(8,128,42)[i] for i in range(8)]
    batch = {key:torch.stack([sample[key] for sample in samples]).to(device) for key in samples[0]}
    times = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(35):  # Five warmups, thirty measured optimizer steps.
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = objective(model(batch['image']), batch['vertices'], batch['valid'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True, foreach=False)
        optimizer.step()
        torch.cuda.synchronize()
        if step >= 5:
            times.append(time.perf_counter()-start)
    print(json.dumps({**environment(device), 'batch_size':8, 'image_size':128,
        'warmup_steps':5, 'measured_steps':30, 'median_step_ms':statistics.median(times)*1000,
        'samples_per_second':8*len(times)/sum(times),
        'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
        'scope':'GPU-resident procedural non-face tensors; excludes disk, decoding, hashing, validation and checkpoint IO'}, indent=2))


if __name__ == '__main__':
    main()
