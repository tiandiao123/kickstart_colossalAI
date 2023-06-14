import torch
import torch.multiprocessing as mp
from colossalai.utils import print_rank_0
from functools import partial

import colossalai
from colossalai.tensor import ProcessGroup, ColoTensor, ColoTensorSpec, ShardSpec, ComputeSpec, ComputePattern
from colossalai.testing import spawn

import torch

def run_dist_tests(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    pg = ProcessGroup(tp_degree=2, dp_degree=2)

    torch.manual_seed(0)
    local_tensor = torch.randn(2, 3, 1).cuda()
    print_rank_0(f"shape {local_tensor.shape}, {local_tensor.data}")

    spec = ColoTensorSpec(pg, ShardSpec(dims=[-1], num_partitions=[pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    t1 = ColoTensor.from_torch_tensor(local_tensor, spec)
    t1 = t1.to_replicate()
    print_rank_0(f"shape {t1.shape}, {t1.data}")

    spec2 = ShardSpec([0], [pg.tp_world_size()])
    t1.set_dist_spec(spec2)
    print_rank_0(f"shape {t1.shape}, {t1.data}")

def test_dist_cases(world_size):
    spawn(run_dist_tests, world_size)

if __name__ == '__main__':
    test_dist_cases(4)