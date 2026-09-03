import unittest
from collections import OrderedDict, defaultdict, deque
from types import SimpleNamespace

import torch

from bench_long_doc_qa import percentile, poisson_arrival_offsets, realized_request_rate, workload_profile
from nanovllm.config import KVCachePolicy
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
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

    def test_poisson_trace_has_requested_length_and_monotonic_offsets(self):
        offsets = poisson_arrival_offsets(100, 2.0, 9)
        self.assertEqual(len(offsets), 100)
        self.assertEqual(offsets[0], 0.0)
        self.assertTrue(all(left < right for left, right in zip(offsets, offsets[1:])))

    def test_default_poisson_trace_preserves_single_request_arrivals(self):
        self.assertEqual(
            poisson_arrival_offsets(10, 2.0, 9),
            poisson_arrival_offsets(10, 2.0, 9, burst_size=1),
        )

    def test_poisson_burst_trace_groups_requests_and_handles_partial_tail(self):
        offsets = poisson_arrival_offsets(10, 2.0, 9, burst_size=4)
        self.assertEqual(len(offsets), 10)
        self.assertEqual(offsets[:4], [0.0] * 4)
        self.assertEqual(len(set(offsets[4:8])), 1)
        self.assertEqual(len(set(offsets[8:])), 1)
        self.assertLess(offsets[3], offsets[4])
        self.assertLess(offsets[7], offsets[8])

    def test_poisson_burst_rate_is_still_expressed_in_requests_per_second(self):
        offsets = poisson_arrival_offsets(100_000, 2.0, 9, burst_size=4)
        self.assertAlmostEqual(realized_request_rate(offsets), 2.0, delta=0.03)

    def test_poisson_burst_size_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "arrival-burst-size"):
            poisson_arrival_offsets(10, 2.0, 9, burst_size=0)

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

    def test_full_pool_prefers_unprotected_gpu_duplicate_over_cpu_lru(self):
        runner = self.runner()
        first = torch.empty(1)
        duplicate = torch.empty(1)
        runner.cpu_prefix_cache[17] = {"block": first, "bytes": 4, "pooled": True}
        runner.cpu_prefix_cache[23] = {"block": duplicate, "bytes": 4, "pooled": True}
        runner.prefix_transfer_metrics["cpu_prefix_kv_bytes"] = 8

        taken, pooled, evicted_hash = runner._take_cpu_prefix_block(
            duplicate.shape, duplicate.dtype, preferred_hashes={23}
        )

        self.assertIs(taken, duplicate)
        self.assertTrue(pooled)
        self.assertEqual(evicted_hash, 23)
        self.assertIn(17, runner.cpu_prefix_cache)
        self.assertEqual(runner.prefix_transfer_metrics["cpu_prefix_cache_preferred_duplicate_eviction_count"], 1)

    def test_full_pool_keeps_protected_gpu_victim_backing_until_last(self):
        runner = self.runner()
        protected = torch.empty(1)
        cpu_only = torch.empty(1)
        runner.cpu_prefix_cache[17] = {"block": protected, "bytes": 4, "pooled": True}
        runner.cpu_prefix_cache[23] = {"block": cpu_only, "bytes": 4, "pooled": True}
        runner.prefix_transfer_metrics["cpu_prefix_kv_bytes"] = 8

        taken, _pooled, evicted_hash = runner._take_cpu_prefix_block(
            cpu_only.shape, cpu_only.dtype, preferred_hashes=set(), protected_hashes={17}
        )

        self.assertIs(taken, cpu_only)
        self.assertEqual(evicted_hash, 23)
        self.assertIn(17, runner.cpu_prefix_cache)
        self.assertEqual(runner.prefix_transfer_metrics["cpu_prefix_cache_preferred_duplicate_eviction_count"], 0)


class _CompletedEvent:
    def __init__(self):
        self.synchronize_count = 0

    def synchronize(self):
        self.synchronize_count += 1

    def query(self):
        return True


class ModelRunnerWritebackBatchTests(unittest.TestCase):
    def test_one_completed_batch_returns_all_writeback_ids(self):
        runner = ModelRunner.__new__(ModelRunner)
        event = _CompletedEvent()
        runner.pending_prefix_writebacks = [{"ids": [7, 8], "done_event": event}]
        runner._record_completed_prefix_writeback = lambda pending: [99]

        result = runner.poll_prefix_writebacks(wait=True)

        self.assertEqual(result, {"completed_ids": [7, 8], "evicted_hashes": [99]})
        self.assertEqual(event.synchronize_count, 1)
        self.assertEqual(runner.pending_prefix_writebacks, [])


class BlockManagerStateTests(unittest.TestCase):
    def test_reports_cpu_gpu_prefix_overlap(self):
        manager = BlockManager(num_blocks=3, block_size=256)
        for h in (10, 20):
            block_id = manager._allocate_block()
            manager.blocks[block_id].update(h, [h])
            manager.hash_to_block_id[h] = block_id
        manager.cpu_hash_to_token_ids = {20: [20], 30: [30]}

        metrics = manager.get_metrics()

        self.assertEqual(metrics["gpu_prefix_cached_block_count"], 2)
        self.assertEqual(metrics["cpu_gpu_duplicate_block_count"], 1)

    def test_sequence_reuses_chained_block_hashes(self):
        old_block_size = Sequence.block_size
        Sequence.block_size = 4
        try:
            seq = Sequence(list(range(9)))
            first = seq.block_hash(0)
            second = seq.block_hash(1)
            self.assertEqual(first, seq.block_hash(0))
            self.assertEqual(second, seq.block_hash(1))
            self.assertEqual(len(seq._block_hashes), 2)
            self.assertEqual(second, BlockManager.compute_hash([4, 5, 6, 7], first))
        finally:
            Sequence.block_size = old_block_size

    def test_cpu_eviction_hints_protect_gpu_lru_victim_front(self):
        manager = BlockManager(num_blocks=3, block_size=256)
        for h in (10, 20, 30):
            block_id = manager._allocate_block()
            manager.blocks[block_id].update(h, [h])
            manager.hash_to_block_id[h] = block_id
            manager.release_blocks([block_id])

        resident, protected = manager.gpu_residency_for_cpu_eviction(1)

        self.assertEqual(resident, {10, 20, 30})
        self.assertEqual(protected, {10})

    def test_evictable_count_tracks_pending_and_activation(self):
        manager = BlockManager(num_blocks=3, block_size=256)
        block_ids = []
        for h in range(3):
            block_id = manager._allocate_block()
            manager.blocks[block_id].update(h, [h])
            manager.release_blocks([block_id])
            block_ids.append(block_id)

        self.assertEqual(manager._available_block_count(), 3)
        entries = [(h, block_id, [h]) for h, block_id in enumerate(block_ids)]
        manager.mark_cpu_writeback_pending(entries)
        self.assertEqual(manager._available_block_count(), 0)
        self.assertTrue(manager.blocks[block_ids[0]].writeback_pending)
        manager.register_cpu_blocks(entries[:1])
        self.assertEqual(manager._available_block_count(), 1)
        manager._activate_inactive_block(block_ids[0])
        self.assertEqual(manager._available_block_count(), 0)


class KVCachePolicyTests(unittest.TestCase):
    def test_legacy_flags_map_to_one_policy(self):
        self.assertIs(KVCachePolicy.from_flags(False, False, True), KVCachePolicy.GPU_RECOMPUTE)
        self.assertIs(KVCachePolicy.from_flags(False, True, True), KVCachePolicy.GPU_LRU)
        self.assertIs(KVCachePolicy.from_flags(True, False, True), KVCachePolicy.CPU_EAGER)
        self.assertIs(KVCachePolicy.from_flags(True, True, False), KVCachePolicy.CPU_EAGER_GPU_LRU)
        self.assertIs(
            KVCachePolicy.from_flags(True, True, False, True),
            KVCachePolicy.CPU_EAGER_GPU_LOOKAHEAD,
        )
        self.assertIs(KVCachePolicy.from_flags(True, True, True), KVCachePolicy.CPU_LAZY_GPU_LRU)
        self.assertIs(
            KVCachePolicy.from_flags(True, True, True, True),
            KVCachePolicy.CPU_LAZY_GPU_LOOKAHEAD,
        )


class SchedulerAwarePrefetchTests(unittest.TestCase):
    def setUp(self):
        self.old_block_size = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.old_block_size

    def test_lookahead_touch_keeps_next_gpu_prefix_and_replacement_uses_lru(self):
        manager = BlockManager(2, 4)
        visible = Sequence([1, 2, 3, 4, 9])
        visible_hash = visible.block_hash(0)
        cold_hash = Sequence.compute_hash([5, 6, 7, 8])
        for block_id, h, tokens in (
            (0, visible_hash, [1, 2, 3, 4]),
            (1, cold_hash, [5, 6, 7, 8]),
        ):
            manager.free_block_ids.remove(block_id)
            manager.blocks[block_id].update(h, tokens)
            manager.hash_to_block_id[h] = block_id
            manager.inactive_block_ids[block_id] = None
            manager.evictable_inactive_block_count += 1

        manager.touch_inactive_blocks([0])
        self.assertEqual(manager.ordered_inactive_block_ids(), [1, 0])

        target = Sequence([10, 11, 12, 13, 9])
        target_hash = target.block_hash(0)
        reservations = manager.plan_prefetch_reservations(
            [(target_hash, target.block_token_ids(0))], set()
        )
        self.assertEqual(reservations[0][2:4], (1, cold_hash))
        manager.commit_prefetch_reservations(reservations)
        self.assertEqual(manager.pending_prefetch_hashes, {target_hash: 1})
        self.assertNotIn(target_hash, manager.hash_to_block_id)
        self.assertEqual(manager.metrics["gpu_lru_eviction_count"], 1)
        self.assertEqual(manager.metrics["gpu_lru_evicted_gpu_only_block_count"], 1)

        manager.complete_prefetch_reservation(target_hash, target.block_token_ids(0), 1)
        self.assertEqual(manager.hash_to_block_id[target_hash], 1)
        manager._activate_inactive_block(1)
        self.assertEqual(manager.metrics["scheduler_prefetch_useful_block_count"], 1)

    def test_n_plus_one_is_hotter_than_n_plus_two(self):
        manager = BlockManager(3, 4)
        first = Sequence([1, 2, 3, 4, 9])
        second = Sequence([5, 6, 7, 8, 9])
        cold_hash = Sequence.compute_hash([10, 11, 12, 13])
        for block_id, h, tokens in (
            (0, first.block_hash(0), first.block_token_ids(0)),
            (1, second.block_hash(0), second.block_token_ids(0)),
            (2, cold_hash, [10, 11, 12, 13]),
        ):
            manager.free_block_ids.remove(block_id)
            manager.blocks[block_id].update(h, tokens)
            manager.hash_to_block_id[h] = block_id
            manager.inactive_block_ids[block_id] = None
            manager.evictable_inactive_block_count += 1

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_scheduler_aware_prefetch = True
        scheduler.waiting = deque([first, second])
        scheduler.max_num_seqs = 8
        scheduler.metrics = defaultdict(int)
        scheduler.block_manager = manager

        def inspect_lru_order(_visible, _decode_reserve):
            self.assertEqual(manager.ordered_inactive_block_ids(), [2, 1, 0])
            return 0, 0

        scheduler._start_v4_prefetch = inspect_lru_order
        scheduler._plan_v4_prefetch([first, second], 0)

        # V3/V4 都使用这一个 LRU backend；V4 只把即将使用的 block touch 到热端。
        self.assertEqual(manager._allocate_block(), 2)

    def test_prefetch_accepts_a_partial_n_plus_one_prefix(self):
        manager = BlockManager(1, 4)
        first = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 9])
        manager.register_cpu_metadata(first.block_hash(0), first.block_token_ids(0))
        manager.register_cpu_metadata(first.block_hash(1), first.block_token_ids(1))

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.waiting = deque([first])
        scheduler.max_num_seqs = 8
        scheduler.scheduler_prefetch_max_blocks = 0
        scheduler.block_manager = manager
        scheduler.metrics = defaultdict(int)

        planned_requests, targets, protected = scheduler._make_v4_prefetch_plan()

        self.assertEqual(planned_requests, 1)
        self.assertEqual(targets, [(first.block_hash(0), first.block_token_ids(0))])
        self.assertEqual(protected, set())
        self.assertEqual(scheduler.metrics["scheduler_prefetch_capacity_rejected_request_count"], 1)

    def test_decode_reserve_can_disable_opportunistic_prefetch(self):
        manager = BlockManager(1, 4)
        seq = Sequence([1, 2, 3, 4, 9])
        manager.register_cpu_metadata(seq.block_hash(0), seq.block_token_ids(0))
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.waiting = deque([seq])
        scheduler.max_num_seqs = 8
        scheduler.scheduler_prefetch_max_blocks = 0
        scheduler.block_manager = manager
        scheduler.metrics = defaultdict(int)

        planned, targets, _protected = scheduler._make_v4_prefetch_plan(decode_reserve=1)

        self.assertEqual(planned, 0)
        self.assertEqual(targets, [])
        self.assertEqual(scheduler.metrics["scheduler_prefetch_capacity_rejected_request_count"], 1)

    def test_pending_prefix_is_not_planned_twice(self):
        manager = BlockManager(2, 4)
        seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 9])
        first_hash = seq.block_hash(0)
        second_hash = seq.block_hash(1)
        manager.pending_prefetch_hashes[first_hash] = 0
        manager.free_block_ids.remove(0)
        manager.blocks[0].replacement_pending = True
        manager.register_cpu_metadata(second_hash, seq.block_token_ids(1))

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.waiting = deque([seq])
        scheduler.max_num_seqs = 8
        scheduler.scheduler_prefetch_max_blocks = 0
        scheduler.block_manager = manager
        scheduler.metrics = defaultdict(int)

        _planned, targets, _protected = scheduler._make_v4_prefetch_plan()

        self.assertEqual(targets, [(second_hash, seq.block_token_ids(1))])


class _BlockManagerStub:
    def __init__(self):
        self.pending = None
        self.unregistered = None

    def mark_cpu_writeback_pending(self, entries):
        self.pending = list(entries)

    def unregister_cpu_blocks(self, hashes):
        self.unregistered = list(hashes)


class SchedulerWritebackProtocolTests(unittest.TestCase):
    def test_naive_v3_ablation_disables_cpu_eviction_hints(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_lazy_cpu_kv_writeback = True
        scheduler.enable_gpu_aware_cpu_eviction = False
        scheduler.lazy_writeback_target_blocks = 40
        scheduler.block_manager = SimpleNamespace(
            gpu_residency_for_cpu_eviction=lambda _window: self.fail("unexpected GPU residency query"),
            mark_cpu_writeback_pending=lambda _entries: None,
        )
        scheduler.pending_prefix_writebacks = {}
        scheduler.pending_writeback_by_block_id = {}
        scheduler.pending_writeback_by_hash = {}
        scheduler.metrics = defaultdict(int)
        captured = []
        scheduler.writeback_prefix_blocks = lambda entries, preferred, protected: (
            captured.append((entries, preferred, protected)) or {"writeback_ids": [7], "evicted_hashes": []}
        )

        accepted = scheduler._submit_prefix_writeback_entries([(11, 3, 256)], release_on_complete=True)

        self.assertEqual(accepted, 1)
        self.assertEqual(captured, [([(11, 3)], None, None)])

    def test_v3_writeback_uses_configured_cpu_eviction_protection(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_lazy_cpu_kv_writeback = True
        scheduler.enable_gpu_aware_cpu_eviction = True
        scheduler.lazy_writeback_target_blocks = 80
        hint_calls = []
        scheduler.block_manager = SimpleNamespace(
            gpu_residency_for_cpu_eviction=lambda window: (
                hint_calls.append(window) or ({11, 12}, {11})
            ),
            mark_cpu_writeback_pending=lambda _entries: None,
        )
        scheduler.pending_prefix_writebacks = {}
        scheduler.pending_writeback_by_block_id = {}
        scheduler.pending_writeback_by_hash = {}
        scheduler.metrics = defaultdict(int)
        scheduler.writeback_prefix_blocks = lambda _entries, _preferred, _protected: {
            "writeback_ids": [7],
            "evicted_hashes": [],
        }

        scheduler._submit_prefix_writeback_entries(
            [(13, 3, [1, 2, 3, 4])],
            release_on_complete=False,
            lazy=True,
        )

        self.assertEqual(hint_calls, [80])

    def test_absolute_lazy_writeback_target_overrides_derived_window(self):
        base = dict(
            max_num_seqs=8,
            max_num_batched_tokens=33152,
            eos=-1,
            kvcache_block_size=256,
            kv_cache_policy=KVCachePolicy.CPU_LAZY_GPU_LRU,
            lazy_writeback_watermark_ratio=0.0,
            enable_gpu_aware_cpu_eviction=True,
            num_kvcache_blocks=227,
            enable_prefix_cache=True,
        )
        self.assertEqual(
            Scheduler(SimpleNamespace(**base, lazy_writeback_target_blocks=60)).lazy_writeback_target_blocks,
            60,
        )
        self.assertEqual(
            Scheduler(SimpleNamespace(**base, lazy_writeback_target_blocks=0)).lazy_writeback_target_blocks,
            130,
        )

    def test_only_accepted_entries_become_pending(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.block_manager = _BlockManagerStub()
        scheduler.pending_prefix_writebacks = {}
        scheduler.pending_writeback_by_block_id = {}
        scheduler.pending_writeback_by_hash = {}
        scheduler.metrics = {
            "pending_prefix_writeback_count": 0,
            "cpu_prefix_cache_evicted_metadata_count": 0,
        }
        scheduler.enable_lazy_cpu_kv_writeback = False
        scheduler.writeback_prefix_blocks = lambda entries, preferred=None, protected=None: {
            "writeback_ids": [41],
            "evicted_hashes": [99],
        }
        entries = [(10, 1, 256), (11, 2, 256)]
        accepted = scheduler._submit_prefix_writeback_entries(entries, release_on_complete=True)
        self.assertEqual(accepted, 1)
        self.assertEqual(scheduler.block_manager.pending, entries[:1])
        self.assertEqual(scheduler.block_manager.unregistered, [99])
        self.assertEqual(list(scheduler.pending_prefix_writebacks), [41])

        self.assertEqual(scheduler.pending_writeback_by_block_id, {1: 41})
        self.assertEqual(scheduler.pending_writeback_by_hash, {10: 41})

    def test_restore_precedes_bounded_pool_writeback_maintenance(self):
        events = []
        seq = SimpleNamespace(
            block_table=[],
            num_blocks=2,
            num_tokens=8,
            num_cached_tokens=0,
            num_scheduled_tokens=0,
            recompute_pending_tokens=0,
            status=None,
        )
        block_manager = SimpleNamespace(
            get_allocate_plan=lambda _seq, _cpu: {"sources": [("cpu", 17, [1, 2, 3, 4])]},
            allocate=lambda _seq, _plan: [(17, 0)],
        )
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.waiting = deque([seq])
        scheduler.running = deque()
        scheduler.max_num_seqs = 1
        scheduler.max_num_batched_tokens = 8
        scheduler.block_size = 4
        scheduler.enable_cpu_kv_offload = True
        scheduler.pending_prefix_writebacks = {}
        scheduler.block_manager = block_manager
        scheduler.restore_prefix_blocks = lambda entries: events.append(("restore", entries))
        scheduler._maintain_lazy_writeback_window_after_allocation = lambda: events.append(("maintain", None))
        scheduler._prepare_v4_prefetch = lambda _seqs: None
        scheduler._poll_prefix_writebacks = lambda wait=False: None
        scheduler.metrics = defaultdict(int)

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [seq])
        self.assertEqual([name for name, _ in events], ["restore", "maintain"])

    def test_v3_after_allocation_maintains_the_fixed_window(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_lazy_cpu_kv_writeback = True
        scheduler.lazy_writeback_target_blocks = 80
        scheduler.block_manager = SimpleNamespace(free_block_count=lambda: 0)
        scheduler.metrics = defaultdict(int)
        calls = []
        scheduler._maintain_lazy_writeback_window = lambda limit=None: calls.append(limit)

        scheduler._maintain_lazy_writeback_window_after_allocation()

        self.assertEqual(calls, [None])
        self.assertEqual(scheduler.metrics["lazy_writeback_after_alloc_trigger_count"], 1)

    def test_v2_v4_prefetch_does_not_enable_v3_window(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_scheduler_aware_prefetch = True
        scheduler.enable_lazy_cpu_kv_writeback = False
        scheduler.waiting = deque()
        scheduler.running = deque()
        scheduler.max_num_seqs = 8
        scheduler.block_size = 4
        scheduler.metrics = defaultdict(int)
        events = []
        scheduler._plan_v4_prefetch = lambda _visible, _reserve: (events.append("prefetch") or (0, 1))
        scheduler._maintain_lazy_writeback_window_after_allocation = lambda: events.append("v3_window")

        scheduler._prepare_v4_prefetch([])

        self.assertEqual(events, ["prefetch"])

    def test_v3_v4_replacement_is_followed_by_the_same_v3_window_check(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_scheduler_aware_prefetch = True
        scheduler.enable_lazy_cpu_kv_writeback = True
        scheduler.waiting = deque()
        scheduler.running = deque()
        scheduler.max_num_seqs = 8
        scheduler.block_size = 4
        scheduler.metrics = defaultdict(int)
        events = []
        scheduler._plan_v4_prefetch = lambda _visible, _reserve: (events.append("prefetch") or (0, 1))
        scheduler._maintain_lazy_writeback_window_after_allocation = lambda: events.append("v3_window")

        scheduler._prepare_v4_prefetch([])

        self.assertEqual(events, ["prefetch", "v3_window"])

    def test_v3_v4_touch_only_does_not_rescan_v3_window(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_scheduler_aware_prefetch = True
        scheduler.enable_lazy_cpu_kv_writeback = True
        scheduler.waiting = deque()
        scheduler.running = deque()
        scheduler.max_num_seqs = 8
        scheduler.block_size = 4
        scheduler.metrics = defaultdict(int)
        events = []
        scheduler._plan_v4_prefetch = lambda _visible, _reserve: (events.append("prefetch") or (0, 0))
        scheduler._maintain_lazy_writeback_window_after_allocation = lambda: events.append("v3_window")

        scheduler._prepare_v4_prefetch([])

        self.assertEqual(events, ["prefetch"])


if __name__ == "__main__":
    unittest.main()
