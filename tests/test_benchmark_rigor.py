import unittest
from collections import OrderedDict, defaultdict
from types import SimpleNamespace

import torch

from bench_long_doc_qa import percentile, poisson_arrival_offsets, workload_profile
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.layers.sampler import Sampler
from nanovllm.sampling_params import SamplingParams


class BenchmarkHelperTests(unittest.TestCase):
    def test_percentile_interpolates_and_handles_singleton(self):
        self.assertEqual(percentile([], 0.99), 0.0)
        self.assertEqual(percentile([7.0], 0.99), 7.0)
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.90), 3.7)

    def test_poisson_trace_is_seeded(self):
        self.assertEqual(
            poisson_arrival_offsets(10, 2.0, 9),
            poisson_arrival_offsets(10, 2.0, 9),
        )
        self.assertNotEqual(
            poisson_arrival_offsets(10, 2.0, 9),
            poisson_arrival_offsets(10, 2.0, 10),
        )

    def test_realized_working_set_counts_only_touched_documents(self):
        args = SimpleNamespace(workload="long_doc_qa", document_length=128)
        profile = workload_profile([1, 1, 3, 1], args, bytes_per_token=4)
        self.assertEqual(profile["unique_document_count"], 2)
        self.assertEqual(profile["access_counts"], {"1": 3, "3": 1})
        self.assertEqual(profile["realized_working_set_tokens"], 256)
        self.assertEqual(profile["reuse_distance"]["count"], 2)


class GreedySamplingTests(unittest.TestCase):
    def test_zero_temperature_is_valid_and_deterministic(self):
        SamplingParams(temperature=0.0)
        logits = torch.tensor([[0.0, 4.0, 2.0], [5.0, -1.0, 1.0]])
        temperatures = torch.zeros(2)
        sampler = Sampler()
        first = sampler(logits, temperatures)
        second = sampler(logits, temperatures)
        self.assertTrue(torch.equal(first, torch.tensor([1, 0])))
        self.assertTrue(torch.equal(first, second))


class CpuPoolHardCapTests(unittest.TestCase):
    @staticmethod
    def runner():
        runner = ModelRunner.__new__(ModelRunner)
        runner.cpu_prefix_pool_free_blocks = []
        runner.cpu_prefix_pool_limit_bytes = 4
        runner.cpu_prefix_cache = OrderedDict()
        runner.prefix_transfer_metrics = defaultdict(int)
        return runner

    def test_full_pool_reuses_lru_live_block(self):
        runner = self.runner()
        block = torch.empty(1)
        runner.cpu_prefix_cache[17] = {"block": block, "bytes": block.numel() * block.element_size(), "pooled": True}
        runner.prefix_transfer_metrics["cpu_prefix_kv_bytes"] = 4
        taken, pooled, evicted_hash = runner._take_cpu_prefix_block(block.shape, block.dtype)
        self.assertIs(taken, block)
        self.assertTrue(pooled)
        self.assertEqual(evicted_hash, 17)
        self.assertEqual(runner.prefix_transfer_metrics["cpu_prefix_pool_on_demand_alloc_count"], 0)

    def test_full_pending_pool_rejects_instead_of_allocating(self):
        runner = self.runner()
        block, pooled, evicted_hash = runner._take_cpu_prefix_block((1,), torch.float32)
        self.assertIsNone(block)
        self.assertFalse(pooled)
        self.assertIsNone(evicted_hash)
        self.assertEqual(runner.prefix_transfer_metrics["cpu_prefix_pool_writeback_rejected_count"], 1)
        self.assertEqual(runner.prefix_transfer_metrics["cpu_prefix_pool_on_demand_alloc_count"], 0)


class _BlockManagerStub:
    def __init__(self):
        self.pending = None
        self.unregistered = None

    def mark_cpu_writeback_pending(self, entries):
        self.pending = list(entries)

    def unregister_cpu_blocks(self, hashes):
        self.unregistered = list(hashes)


class SchedulerWritebackProtocolTests(unittest.TestCase):
    def test_only_accepted_entries_become_pending(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.block_manager = _BlockManagerStub()
        scheduler.pending_prefix_writebacks = {}
        scheduler.metrics = {
            "pending_prefix_writeback_count": 0,
            "cpu_prefix_cache_evicted_metadata_count": 0,
        }
        scheduler.writeback_prefix_blocks = lambda entries: {
            "writeback_ids": [41],
            "evicted_hashes": [99],
        }
        entries = [(10, 1, 256), (11, 2, 256)]
        accepted = scheduler._submit_prefix_writeback_entries(entries, release_on_complete=True)
        self.assertEqual(accepted, 1)
        self.assertEqual(scheduler.block_manager.pending, entries[:1])
        self.assertEqual(scheduler.block_manager.unregistered, [99])
        self.assertEqual(list(scheduler.pending_prefix_writebacks), [41])


if __name__ == "__main__":
    unittest.main()
