from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.enable_cpu_kv_offload = config.enable_cpu_kv_offload
        self.swap_out = None
        self.swap_in = None
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size, config.enable_prefix_cache)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.reset_metrics()

    def reset_metrics(self):
        self.metrics = {
            "preemption_count": 0,
            "recomputed_token_count": 0,
            "prefill_token_count": 0,
            "prefix_cache_lookup_count": 0,
            "prefix_cache_reused_token_count": 0,
        }

    def get_metrics(self):
        return {
            "preemption_count": self.metrics["preemption_count"],
            "recomputed_token_count": self.metrics["recomputed_token_count"],
            "prefill_token_count": self.metrics["prefill_token_count"],
            "prefix_cache_lookup_count": self.metrics["prefix_cache_lookup_count"],
            "prefix_cache_reused_token_count": self.metrics["prefix_cache_reused_token_count"],
        }

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            needs_swap_in = False
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                if seq.is_swapped:
                    assert self.enable_cpu_kv_offload and self.swap_in is not None
                    num_cached_blocks = 0
                    num_tokens = seq.num_tokens - seq.cpu_cached_tokens
                    needs_swap_in = True
                else:
                    num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
                    self.metrics["prefix_cache_lookup_count"] += 1
                    self.metrics["prefix_cache_reused_token_count"] += num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
                if needs_swap_in:
                    self.swap_in(seq)
                    seq.num_cached_tokens = seq.cpu_cached_tokens
                    seq.cpu_cached_tokens = 0
                    seq.is_swapped = False
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            self.metrics["prefill_token_count"] += seq.num_scheduled_tokens
            if seq.recompute_pending_tokens:
                recomputed = min(seq.num_scheduled_tokens, seq.recompute_pending_tokens)
                self.metrics["recomputed_token_count"] += recomputed
                seq.recompute_pending_tokens -= recomputed
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True         # 优先做prefill，且只做prefill

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):   # 显存塞不下了
                if self.running:    # 抢占队尾，放入waiting队列
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))   # 重新插入队头，保证纯粹的FCFS
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        lost_tokens = seq.num_cached_tokens
        if lost_tokens:
            self.metrics["preemption_count"] += 1
            seq.num_preemptions += 1
            if self.enable_cpu_kv_offload:
                assert self.swap_out is not None
                self.swap_out(seq)
                seq.is_swapped = True
                seq.cpu_cached_tokens = lost_tokens
            else:
                seq.recompute_pending_tokens += lost_tokens
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
