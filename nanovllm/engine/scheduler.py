from collections import deque
from dataclasses import dataclass
import math
from time import perf_counter

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus

from nanovllm.engine.block_manager import BlockManager

@dataclass(slots=True)
class PendingPrefixWriteback:
    prefix_hash: int
    block_id: int
    token_ids: list[int]
    release_on_complete: bool
    lazy: bool


@dataclass(slots=True)
class PendingPrefixReplacement:
    # Scheduler 只保存完成 event 后更新元数据所需的信息；KV tensor 和 CUDA event
    # 由 ModelRunner 持有。wrote_back_victim=False 表示 victim 原本已有 CPU backing。
    target_hash: int
    target_tokens: list[int]
    block_id: int
    victim_hash: int | None
    victim_tokens: list[int] | None
    wrote_back_victim: bool



class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.kv_cache_policy = config.kv_cache_policy
        self.enable_cpu_kv_offload = self.kv_cache_policy.cpu_offload
        self.enable_lazy_cpu_kv_writeback = self.kv_cache_policy.lazy_writeback
        self.enable_gpu_aware_cpu_eviction = config.enable_gpu_aware_cpu_eviction
        self.enable_scheduler_aware_prefetch = self.kv_cache_policy.scheduler_aware
        self.scheduler_prefetch_max_blocks = getattr(config, "scheduler_prefetch_max_blocks", 0)
        target_alloc_blocks = math.ceil(self.max_num_batched_tokens / self.block_size)
        # V3 memory-aware selective writeback 的目标窗口来自 vLLM lazy offload 思路：
        # 一轮 scheduler step 最多新增 target_alloc_blocks 个 KV blocks，再乘 watermark 留冗余。
        # 这里维护的是 inactive LRU 中“已 CPU-backed 或正在写回”的 victim 数量，
        # 而不是一看到 request prefill 完就把它的所有 prefix 都搬到 CPU。
        self.lazy_writeback_target_blocks = config.lazy_writeback_target_blocks or math.ceil(
            target_alloc_blocks * (1 + config.lazy_writeback_watermark_ratio)
        )
        # V1 prefix offload 通过 LLMEngine 注入 ModelRunner 侧的 copy 回调。
        self.writeback_prefix_blocks = None
        self.restore_prefix_blocks = None
        self.poll_prefix_writebacks = None
        self.replace_prefix_blocks = None
        self.poll_prefix_replacements = None
        # key 是异步 D2H writeback id，value 是本次写回涉及的 prefix blocks；
        # pending 期间这些 GPU blocks 仍保存有效 KV，但不能回到 free list。
        self.pending_prefix_writebacks = {}
        self.pending_writeback_by_block_id: dict[int, int] = {}
        self.pending_writeback_by_hash: dict[int, int] = {}
        # replacement id -> 本次“victim 写回 + target 预取”的元数据事务。
        # slot 在 pending 期间既不属于旧 victim，也不会提前暴露成 target GPU hit。
        self.pending_prefix_replacements: dict[int, PendingPrefixReplacement] = {}
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            config.enable_prefix_cache,
            # 开关含义：V1 关闭 inactive GPU LRU；V2 打开 inactive GPU LRU。
            self.kv_cache_policy.gpu_lru,
        )
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
            "gpu_prefix_miss_request_count": 0,
            # CPU prefix cache 指标用于区分：GPU 直接命中 vs CPU 命中后 restore。
            "cpu_prefix_cache_hit_count": 0,
            "cpu_prefix_cache_restored_token_count": 0,
            # sync_swapin 只统计 demand path 上的 GPU miss + CPU hit + 同步 H2D。
            # V3 prefetch 成功后如果已经变成 GPU hit，不应该计入这里。
            "cpu_sync_swapin_request_count": 0,
            "cpu_sync_swapin_block_count": 0,
            "cpu_sync_swapin_token_count": 0,
            "pending_prefix_writeback_count": 0,
            "lazy_writeback_target_block_count": self.lazy_writeback_target_blocks if self.enable_lazy_cpu_kv_writeback else 0,
            "lazy_writeback_completed_block_count": 0,
            "cpu_prefix_cache_evicted_metadata_count": 0,
            "lazy_writeback_after_alloc_check_count": 0,
            "lazy_writeback_after_alloc_trigger_count": 0,
            "lazy_writeback_after_alloc_skip_count": 0,
            "lazy_writeback_maintain_call_count": 0,
            "lazy_writeback_maintain_wall_sec": 0.0,
            "lazy_writeback_pending_hash_wall_sec": 0.0,
            "lazy_writeback_select_wall_sec": 0.0,
            "lazy_writeback_submit_wall_sec": 0.0,
            "scheduler_visible_request_count_sum": 0,
            "scheduler_visible_request_count_max": 0,
            "scheduler_prefetch_planned_request_count": 0,
            "scheduler_prefetch_planned_block_count": 0,
            "scheduler_prefetch_completed_block_count": 0,
            "scheduler_prefetch_targeted_eviction_count": 0,
            "scheduler_prefetch_targeted_writeback_count": 0,
            "scheduler_prefetch_wait_count": 0,
            "scheduler_prefetch_wait_wall_sec": 0.0,
            "scheduler_victim_no_visible_next_use_count": 0,
            "scheduler_victim_future_next_use_count": 0,
        }
        self.block_manager.reset_metrics()

    def get_metrics(self):
        metrics = {
            "preemption_count": self.metrics["preemption_count"],
            "recomputed_token_count": self.metrics["recomputed_token_count"],
            "prefill_token_count": self.metrics["prefill_token_count"],
            "prefix_cache_lookup_count": self.metrics["prefix_cache_lookup_count"],
            "prefix_cache_reused_token_count": self.metrics["prefix_cache_reused_token_count"],
            "gpu_prefix_miss_request_count": self.metrics["gpu_prefix_miss_request_count"],
            "cpu_prefix_cache_hit_count": self.metrics["cpu_prefix_cache_hit_count"],
            "cpu_prefix_cache_restored_token_count": self.metrics["cpu_prefix_cache_restored_token_count"],
            "cpu_sync_swapin_request_count": self.metrics["cpu_sync_swapin_request_count"],
            "cpu_sync_swapin_block_count": self.metrics["cpu_sync_swapin_block_count"],
            "cpu_sync_swapin_token_count": self.metrics["cpu_sync_swapin_token_count"],
            # 返回实时 pending 数，避免 reset 后 metrics 里的旧值和实际状态不一致。
            "pending_prefix_writeback_count": len(self.pending_prefix_writebacks),
            "lazy_writeback_target_block_count": self.lazy_writeback_target_blocks if self.enable_lazy_cpu_kv_writeback else 0,
            "lazy_writeback_completed_block_count": self.metrics["lazy_writeback_completed_block_count"],
            "cpu_prefix_cache_evicted_metadata_count": self.metrics["cpu_prefix_cache_evicted_metadata_count"],
            "lazy_writeback_after_alloc_check_count": self.metrics["lazy_writeback_after_alloc_check_count"],
            "lazy_writeback_after_alloc_trigger_count": self.metrics["lazy_writeback_after_alloc_trigger_count"],
            "lazy_writeback_after_alloc_skip_count": self.metrics["lazy_writeback_after_alloc_skip_count"],
            "lazy_writeback_maintain_call_count": self.metrics["lazy_writeback_maintain_call_count"],
            "lazy_writeback_maintain_wall_sec": self.metrics["lazy_writeback_maintain_wall_sec"],
            "lazy_writeback_pending_hash_wall_sec": self.metrics["lazy_writeback_pending_hash_wall_sec"],
            "lazy_writeback_select_wall_sec": self.metrics["lazy_writeback_select_wall_sec"],
            "lazy_writeback_submit_wall_sec": self.metrics["lazy_writeback_submit_wall_sec"],
            "scheduler_visible_request_count_sum": self.metrics["scheduler_visible_request_count_sum"],
            "scheduler_visible_request_count_max": self.metrics["scheduler_visible_request_count_max"],
            "scheduler_prefetch_planned_request_count": self.metrics["scheduler_prefetch_planned_request_count"],
            "scheduler_prefetch_planned_block_count": self.metrics["scheduler_prefetch_planned_block_count"],
            "scheduler_prefetch_completed_block_count": self.metrics["scheduler_prefetch_completed_block_count"],
            "scheduler_prefetch_targeted_eviction_count": self.metrics["scheduler_prefetch_targeted_eviction_count"],
            "scheduler_prefetch_targeted_writeback_count": self.metrics["scheduler_prefetch_targeted_writeback_count"],
            "scheduler_prefetch_wait_count": self.metrics["scheduler_prefetch_wait_count"],
            "scheduler_prefetch_wait_wall_sec": self.metrics["scheduler_prefetch_wait_wall_sec"],
            "scheduler_victim_no_visible_next_use_count": self.metrics["scheduler_victim_no_visible_next_use_count"],
            "scheduler_victim_future_next_use_count": self.metrics["scheduler_victim_future_next_use_count"],
            "pending_prefix_replacement_count": len(self.pending_prefix_replacements),
        }
        metrics.update(self.block_manager.get_metrics(
            self.lazy_writeback_target_blocks if self.enable_lazy_cpu_kv_writeback else 0
        ))
        metrics["victim_window_safe_or_pending_block_count"] = (
            self.block_manager.victim_window_safe_or_pending_count(self.lazy_writeback_target_blocks)
            if self.enable_lazy_cpu_kv_writeback else 0
        )
        return metrics

    def _visible_waiting(self):
        # 第一版不偷看未来 trace，也不重排请求；只看已经到达的 FCFS 队头窗口。
        return list(self.waiting)[:self.max_num_seqs]

    def _update_v4_victim_order(self, record_window: bool = True):
        if not self.enable_scheduler_aware_prefetch:
            return
        visible = self._visible_waiting()
        if record_window:
            self.metrics["scheduler_visible_request_count_sum"] += len(visible)
            self.metrics["scheduler_visible_request_count_max"] = max(
                self.metrics["scheduler_visible_request_count_max"], len(visible)
            )
        self.block_manager.update_victim_order(visible)

    def _make_v4_prefetch_plan(self):
        """按 FCFS 接纳能完整放入 GPU 的请求，并返回其中尚在 CPU 的 prefix。"""
        # protected_gpu：已接纳请求将直接命中的 GPU blocks，后续选 victim 必须避开。
        # target_hashes：已经 pending 或本轮已经选中的 prefix，用于跨请求共享去重。
        # capacity_used：这些请求未来真正还需占用的新 slot 数量。
        protected_gpu = set()
        targets = []
        target_hashes = set(self.block_manager.pending_prefetch_hashes)
        capacity_used = 0
        planned_requests = 0

        for seq in self._visible_waiting():
            if seq.block_table:
                continue
            sources = []
            seq_gpu = set()
            # 为请求构造连续 prefix 来源：[gpu | pending | cpu]。遇到第一个 miss
            # 就停止，剩余部分都必须走正常 prefill，不能跳跃命中后面的 block。
            for block_idx in range(max(seq.num_blocks - 1, 0)):
                tokens = seq.block_token_ids(block_idx)
                h = seq.block_hash(block_idx)
                block_id = self.block_manager.hash_to_block_id.get(h)
                if block_id is not None and self.block_manager.blocks[block_id].token_ids == tokens:
                    sources.append("gpu")
                    seq_gpu.add(block_id)
                elif h in self.block_manager.pending_prefetch_hashes:
                    sources.append("pending")
                elif self.block_manager.has_cpu_block(h, tokens):
                    sources.append("cpu")
                else:
                    break

            new_targets = []
            # 只预取仍在 CPU 的连续 prefix；GPU hit 和已经 pending 的 prefix 不重复提交。
            # 多个 visible requests 共享同一 hash 时，本轮也只为它保留一个 slot。
            for block_idx, source in enumerate(sources):
                if source != "cpu":
                    continue
                h = seq.block_hash(block_idx)
                if h not in target_hashes:
                    new_targets.append((h, seq.block_token_ids(block_idx)))

            # 精确容量口径：
            #   新 slot = 当前请求的 uncached suffix + 本轮首次出现的 CPU targets。
            # GPU/pending prefix 已有物理 slot，不重复计数；共享 target 只计一次。
            suffix_blocks = seq.num_blocks - len(sources)
            needed = suffix_blocks + len(new_targets)
            candidate_protected = protected_gpu | seq_gpu
            if capacity_used + needed > self.block_manager._available_block_count(candidate_protected):
                # 保持 FCFS：队头请求完整放不下时就停止，不能越过它预取后面的请求。
                break
            protected_gpu = candidate_protected
            capacity_used += needed
            planned_requests += 1
            targets.extend(new_targets)
            target_hashes.update(h for h, _tokens in new_targets)

        if self.scheduler_prefetch_max_blocks:
            # max_num_seqs 限制可见请求数；这里的 block budget 只进一步限制传输量。
            # 即使配置为 0（不限），上面的完整请求容量检查仍保证不会超配 GPU slot。
            targets = targets[:self.scheduler_prefetch_max_blocks]
        return planned_requests, targets, protected_gpu

    def _start_v4_prefetch(self):
        if not self.enable_scheduler_aware_prefetch or not self.waiting:
            return 0, 0
        planned_requests, targets, protected_gpu = self._make_v4_prefetch_plan()
        if not targets:
            return 0, 0
        reservations = self.block_manager.plan_prefetch_reservations(targets, protected_gpu)
        if not reservations:
            return 0, 0
        assert self.replace_prefix_blocks is not None
        # ModelRunner 可能因为固定 CPU pool 暂时拿不到 victim 写回 buffer，只接受前缀子集。
        # 因此必须先提交 CUDA，再仅 commit 实际返回 replacement id 的 reservations。
        result = self.replace_prefix_blocks(reservations)
        replacement_ids = result["replacement_ids"] if isinstance(result, dict) else result
        accepted = reservations[:len(replacement_ids)]
        self.block_manager.commit_prefetch_reservations(accepted)
        for replacement_id, reservation in zip(replacement_ids, accepted):
            h, tokens, block_id, victim_hash, victim_tokens, needs_writeback = reservation
            self.pending_prefix_replacements[replacement_id] = PendingPrefixReplacement(
                h, tokens, block_id, victim_hash, victim_tokens, needs_writeback
            )
        # 为写回 victim 腾 CPU buffer 时，ModelRunner 可能淘汰别的 CPU cache entry；
        # Scheduler 同步删除镜像 metadata，避免以后误判成 CPU hit。
        if isinstance(result, dict) and result.get("evicted_hashes"):
            evicted = result["evicted_hashes"]
            self.block_manager.unregister_cpu_blocks(evicted)
            self.metrics["cpu_prefix_cache_evicted_metadata_count"] += len(evicted)
        self.metrics["scheduler_prefetch_planned_request_count"] += planned_requests
        self.metrics["scheduler_prefetch_planned_block_count"] += len(accepted)
        self.metrics["scheduler_prefetch_targeted_eviction_count"] += sum(r[3] is not None for r in accepted)
        self.metrics["scheduler_prefetch_targeted_writeback_count"] += sum(r[5] for r in accepted)
        for reservation in accepted:
            block_id, victim_hash = reservation[2], reservation[3]
            if victim_hash is None:
                continue
            key = (
                "scheduler_victim_future_next_use_count"
                if victim_hash in self.block_manager.visible_next_use
                else "scheduler_victim_no_visible_next_use_count"
            )
            self.metrics[key] += 1
        return sum(r[5] for r in accepted), len(accepted)

    def _plan_v4_prefetch(self):
        """在 demand 之后临时使用 OPT 规划 replacement，提交后立即恢复 LRU。"""
        self._update_v4_victim_order()
        targeted_writebacks, replacement_count = self._start_v4_prefetch()
        # OPT 的任务到 slot reservation 提交就结束了。下面的 V3 后台写回
        # 必须继续保护 LRU 前沿，下一轮 demand allocation 也会按 LRU 驱逐。
        self.block_manager.clear_victim_order()
        return targeted_writebacks, replacement_count

    def _poll_prefix_replacements(self, wait: bool = False):
        if not self.pending_prefix_replacements:
            return
        assert self.poll_prefix_replacements is not None
        result = self.poll_prefix_replacements(wait)
        completed_ids = result["completed_ids"] if isinstance(result, dict) else result
        for replacement_id in completed_ids:
            pending = self.pending_prefix_replacements.pop(replacement_id)
            # 一个 final_event 同时封住 D2H 和 H2D。只有它完成后才能：
            # 1. 登记 victim 的新 CPU backing；2. 将 target 正式公开为 GPU-resident。
            if pending.wrote_back_victim:
                self.block_manager.register_cpu_metadata(pending.victim_hash, pending.victim_tokens)
            self.block_manager.complete_prefetch_reservation(
                pending.target_hash, pending.target_tokens, pending.block_id
            )
            self.metrics["scheduler_prefetch_completed_block_count"] += 1
        if isinstance(result, dict) and result.get("evicted_hashes"):
            evicted = result["evicted_hashes"]
            self.block_manager.unregister_cpu_blocks(evicted)
            self.metrics["cpu_prefix_cache_evicted_metadata_count"] += len(evicted)


    def _pending_writeback_block_ids(self) -> set[int]:
        # 这些 block 的 D2H 已经提交但尚未完成。GPU KV 仍有效，
        # 但不能被 allocator 覆盖；否则 copy stream 可能读到被新请求改写后的内容。
        return set(self.pending_writeback_by_block_id)

    def _pending_writeback_hashes(self) -> set[int]:
        # 用 hash 去重：同一个 prefix 已经在写回队列里时，不再重复提交 D2H。
        return set(self.pending_writeback_by_hash)

    def _mark_pending_writeback_release(self, block_ids: list[int]):
        # request finish/preempt 时，如果某些 block 还在 D2H pending，
        # 不能立刻 release；只把 pending 记录标成“完成后再释放”。
        for block_id in block_ids:
            writeback_id = self.pending_writeback_by_block_id.get(block_id)
            if writeback_id is not None:
                self.pending_prefix_writebacks[writeback_id].release_on_complete = True

    def _decode_tail_tokens_without_prefix_backing(self, seq: Sequence) -> int:
        # V1 invariant: prompt prefill 完成后，完整 prefix blocks 已经 CPU_RESIDENT 或 WRITEBACK_PENDING。
        # decode 阶段新产生的 KV 暂不进入 prefix cache，因此抢占时只需要重算最后一个未满 block。
        return seq.num_cached_tokens % self.block_size

    def _submit_prefix_writeback_entries(self, entries, release_on_complete: bool, lazy: bool = False) -> int:
        if not entries:
            return 0
        assert self.writeback_prefix_blocks is not None
        preferred_cpu_evictions = None
        protected_cpu_evictions = None
        if self.enable_lazy_cpu_kv_writeback and self.enable_gpu_aware_cpu_eviction:
            gpu_resident, protected = self.block_manager.gpu_residency_for_cpu_eviction(
                self.lazy_writeback_target_blocks
            )
            # Keep CPU backing for the GPU LRU victim front.  Copies for active/MRU
            # and other window-external GPU blocks are redundant and leave first.
            preferred_cpu_evictions = gpu_resident - protected
            protected_cpu_evictions = protected
        writeback_result = self.writeback_prefix_blocks(
            [(h, block_id) for h, block_id, _tokens in entries],
            preferred_cpu_evictions,
            protected_cpu_evictions,
        )
        if isinstance(writeback_result, dict):
            writeback_ids = writeback_result["writeback_ids"]
            evicted_hashes = writeback_result.get("evicted_hashes", [])
        else:
            writeback_ids = writeback_result
            evicted_hashes = []
        accepted_entries = entries[:len(writeback_ids)]
        # Only accepted D2H copies are protected. A hard-capped CPU pool may
        # reject the tail when every reserved block is already pending.
        self.block_manager.mark_cpu_writeback_pending(accepted_entries)
        for writeback_id, entry in zip(writeback_ids, accepted_entries):
            h, block_id, token_ids = entry
            pending = PendingPrefixWriteback(h, block_id, token_ids, release_on_complete, lazy)
            self.pending_prefix_writebacks[writeback_id] = pending
            self.pending_writeback_by_block_id[block_id] = writeback_id
            self.pending_writeback_by_hash[h] = writeback_id
        if evicted_hashes:
            self.block_manager.unregister_cpu_blocks(evicted_hashes)
            self.metrics["cpu_prefix_cache_evicted_metadata_count"] += len(evicted_hashes)
        self.metrics["pending_prefix_writeback_count"] = len(self.pending_prefix_writebacks)
        return len(writeback_ids)

    def _start_eager_prefix_writeback(self, seq: Sequence, release_on_complete: bool):
        # V1/V2 专用入口：按 request 粒度 eager 备份完整 prompt prefix。
        # V3 lazy 模式不会走这里，因为它不想让 active request 无脑占一份 CPU backing。
        if not self.enable_cpu_kv_offload or seq.eager_prefix_writeback_done:
            return
        entries = self.block_manager.prefix_entries(seq)
        pending_hashes = self._pending_writeback_hashes()
        entries = [
            (h, block_id, tokens)
            for h, block_id, tokens in entries
            if h not in pending_hashes and not self.block_manager.has_cpu_block(h, tokens)
        ]
        if not entries:
            seq.eager_prefix_writeback_done = True
            return
        # V1/V2 eager writeback：prefill 完成后立刻把完整 prefix 异步 D2H 到 CPU。
        # V3 lazy writeback 会跳过这里，只在 inactive 安全窗口不足时选择性写回。
        self._submit_prefix_writeback_entries(entries, release_on_complete, lazy=False)
        seq.eager_prefix_writeback_done = True

    def _maintain_lazy_writeback_window(self, max_entries: int | None = None):
        # V3 专用入口：按 cache 压力维护 inactive LRU 的 CPU-backed victim 窗口。
        # profile 指标只拆 Python 热路径，不改变异步 D2H 的执行语义。
        if not self.enable_lazy_cpu_kv_writeback:
            return
        maintain_start = perf_counter()
        self.metrics["lazy_writeback_maintain_call_count"] += 1

        # V3 的触发点只放在真正发生 allocation 之后：
        # free blocks 跌破 safety window，才扫描 inactive LRU 前沿补 CPU-backed victim。
        pending_start = perf_counter()
        pending_hashes = self._pending_writeback_hashes()
        self.metrics["lazy_writeback_pending_hash_wall_sec"] += perf_counter() - pending_start

        select_start = perf_counter()
        entries = self.block_manager.select_lazy_writeback_entries(
            self.lazy_writeback_target_blocks,
            pending_hashes,
            max_entries,
        )
        self.metrics["lazy_writeback_select_wall_sec"] += perf_counter() - select_start

        # lazy writeback 的 block 已经 inactive，所以 release_on_complete=False；
        # D2H 完成后只是把它标成 CPU-backed victim，不需要再释放 request 引用。
        submit_start = perf_counter()
        self._submit_prefix_writeback_entries(entries, release_on_complete=False, lazy=True)
        self.metrics["lazy_writeback_submit_wall_sec"] += perf_counter() - submit_start
        self.metrics["lazy_writeback_maintain_wall_sec"] += perf_counter() - maintain_start

    def _maintain_lazy_writeback_window_after_allocation(self, max_entries: int | None = None):
        if not self.enable_lazy_cpu_kv_writeback:
            return
        self.metrics["lazy_writeback_after_alloc_check_count"] += 1
        # 真正 free 的 block 是零成本容量；inactive block 仍保存旧 KV，覆盖它才算驱逐。
        # 只有 free 水位不足时才启动 lazy writeback，避免每个 decode step 都扫 LRU。
        if self.block_manager.free_block_count() >= self.lazy_writeback_target_blocks:
            self.metrics["lazy_writeback_after_alloc_skip_count"] += 1
            return
        self.metrics["lazy_writeback_after_alloc_trigger_count"] += 1
        self._maintain_lazy_writeback_window(max_entries)

    def _v4_background_writeback_budget(self, targeted_writebacks: int):
        # 定向 replacement 比后台安全窗口优先。显式设置 block budget 时，先扣掉
        # 本轮为 victim 做的 D2H，余量才交给 V3 lazy writeback。
        if not self.scheduler_prefetch_max_blocks:
            return None
        return max(self.scheduler_prefetch_max_blocks - targeted_writebacks, 0)

    # 回收一下写完的 prefix blocks，用于后续使用。wait=True 时阻塞等待
    def _poll_prefix_writebacks(self, wait: bool = False):
        if not self.enable_cpu_kv_offload or not self.pending_prefix_writebacks:
            return
        assert self.poll_prefix_writebacks is not None
        poll_result = self.poll_prefix_writebacks(wait)
        if isinstance(poll_result, dict):
            completed_ids = poll_result["completed_ids"]
            evicted_hashes = poll_result.get("evicted_hashes", [])
        else:
            completed_ids = poll_result
            evicted_hashes = []
        for writeback_id in completed_ids:
            pending = self.pending_prefix_writebacks.pop(writeback_id)
            entry = (pending.prefix_hash, pending.block_id, pending.token_ids)
            self.pending_writeback_by_block_id.pop(pending.block_id, None)
            if self.pending_writeback_by_hash.get(pending.prefix_hash) == writeback_id:
                self.pending_writeback_by_hash.pop(pending.prefix_hash)
            self.block_manager.register_cpu_blocks([entry])
            if pending.lazy:
                self.metrics["lazy_writeback_completed_block_count"] += 1
            if pending.release_on_complete:
                self.block_manager.release_blocks([pending.block_id])
        if evicted_hashes:
            self.block_manager.unregister_cpu_blocks(evicted_hashes)
            self.metrics["cpu_prefix_cache_evicted_metadata_count"] += len(evicted_hashes)
        self.metrics["pending_prefix_writeback_count"] = len(self.pending_prefix_writebacks)

    def is_finished(self):
        if not self.waiting and not self.running:
            # 所有 request 都结束后，等待最后一批主动写回完成，保证 CPU cache 元数据落稳。
            if self.enable_scheduler_aware_prefetch:
                self._poll_prefix_replacements(wait=True)
            self._poll_prefix_writebacks(wait=True)
            return True
        if self.enable_scheduler_aware_prefetch:
            self._poll_prefix_replacements()
        self._poll_prefix_writebacks()
        return False

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # 每轮先非阻塞 query 已提交的 CUDA events
        # 已完成 replacement 会在这里公开 target，writeback 则补齐 CPU backing metadata。
        v4 = getattr(self, "enable_scheduler_aware_prefetch", False)
        if v4:
            self._poll_prefix_replacements()
        self._poll_prefix_writebacks()
        scheduled_seqs = []
        num_batched_tokens = 0
        allocated_this_tick = False

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            restore_entries = []
            if not seq.block_table:
                if v4 and self.block_manager.has_pending_prefetch(seq):
                    # 请求已经到 FCFS 队头，但它依赖的异步 H2D 还没完成：只在这个关键路径
                    # 同步等待 final_event；普通 scheduler tick 始终只做非阻塞 query。
                    self.metrics["scheduler_prefetch_wait_count"] += 1
                    wait_start = perf_counter()
                    self._poll_prefix_replacements(wait=True)
                    self.metrics["scheduler_prefetch_wait_wall_sec"] += perf_counter() - wait_start
                # 读入 request 时按最长连续 prefix 制定分配计划：
                # GPU hit 直接复用 block；GPU miss + CPU hit 先分配 GPU block，
                # 再同步 restore；GPU/CPU 都 miss 的尾部正常 prefill。
                plan = self.block_manager.get_allocate_plan(seq, self.enable_cpu_kv_offload)
                if plan is None and v4 and self.pending_prefix_replacements:
                    # 容量可能只是暂时锁在 REPLACEMENT_PENDING slot 中。等待已有 replacement
                    # 后重算一次 allocation plan，避免把“尚未完成”误判成真正显存不足。
                    self._poll_prefix_replacements(wait=True)
                    plan = self.block_manager.get_allocate_plan(seq, self.enable_cpu_kv_offload)
                if plan is None and self.pending_prefix_writebacks:
                    # 如果显存看似不够，先等待已发起的 writeback 完成
                    self._poll_prefix_writebacks(wait=True)
                    plan = self.block_manager.get_allocate_plan(seq, self.enable_cpu_kv_offload)
                if plan is None:
                    break   # 如果还是不够，放弃本轮 prefill，等待下一轮调度。
                num_cached_blocks = len(plan["sources"])
                gpu_hits = sum(1 for source, _h, _tokens in plan["sources"] if source == "gpu")
                cpu_hits = num_cached_blocks - gpu_hits
                cacheable_prefix_blocks = max(seq.num_blocks - 1, 0)
                # GPU hit 和 CPU hit 都可以跳过对应 prefix prefill；区别是 CPU hit 需要先 H2D restore。
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
                self.metrics["prefix_cache_lookup_count"] += 1
                if gpu_hits < cacheable_prefix_blocks:
                    self.metrics["gpu_prefix_miss_request_count"] += 1
                self.metrics["prefix_cache_reused_token_count"] += gpu_hits * self.block_size
                self.metrics["cpu_prefix_cache_hit_count"] += cpu_hits
                self.metrics["cpu_prefix_cache_restored_token_count"] += cpu_hits * self.block_size
                if cpu_hits:
                    # 这里统计的是关键路径上的 demand sync swapin：GPU miss、CPU hit，且必须立刻 H2D 才能继续 prefill。
                    # V3 如果提前 prefetch 成功，request 到这里应表现为 GPU hit，不能再计入 sync_swapin。
                    self.metrics["cpu_sync_swapin_request_count"] += 1
                    self.metrics["cpu_sync_swapin_block_count"] += cpu_hits
                    self.metrics["cpu_sync_swapin_token_count"] += cpu_hits * self.block_size
            else:
                plan = None
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                restore_entries = self.block_manager.allocate(seq, plan)    # GPU hit 直接引用，CPU hit 先分配 GPU block，miss/全新请求分配空 block
                allocated_this_tick = True
                if restore_entries:
                    assert self.restore_prefix_blocks is not None
                    self.restore_prefix_blocks(restore_entries) # 同步 H2D 后再继续 prefill 剩余 suffix。
                # restore plan 是按 allocation 前的 CPU cache 快照生成的。必须先消费它，
                # 再允许 bounded CPU pool 的 lazy writeback 淘汰 LRU backing；否则同轮
                # writeback 可能复用 restore_entries 仍要读取的 CPU block。
                if not v4:
                    self._maintain_lazy_writeback_window_after_allocation()
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
            if v4:
                # demand allocation 保持 V3 LRU；完成后才查看剩余 waiting window，
                # 用临时 OPT 顺序提交下一批异步 replacement。
                targeted_writebacks, replacement_count = self._plan_v4_prefetch()
                if allocated_this_tick or replacement_count:
                    self._maintain_lazy_writeback_window_after_allocation(
                        self._v4_background_writeback_budget(targeted_writebacks)
                    )
            return scheduled_seqs, True         # 优先做prefill，且只做prefill

        # decode
        # decode 队列左边是高优先级老请求。
        # 每轮从左边取请求 decode。
        # 如果显存不够，就从右边抢占低优先级请求。
        # 调度成功的请求再放回左边，维持 FCFS。
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
                allocated = self.block_manager.may_append(seq)
                if allocated:
                    allocated_this_tick = True
                    if not v4:
                        self._maintain_lazy_writeback_window_after_allocation()
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))   # 重新插入队头，保证纯粹的FCFS
        if v4:
            # decode tick 同样利用本轮计算时间预取 waiting requests；优先级仍是
            # 当前 demand > 定向 replacement > 剩余 budget 的 V3 后台 writeback。
            targeted_writebacks, replacement_count = self._plan_v4_prefetch()
            if allocated_this_tick or replacement_count:
                self._maintain_lazy_writeback_window_after_allocation(
                    self._v4_background_writeback_budget(targeted_writebacks)
                )
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        lost_tokens = seq.num_cached_tokens
        protected = set()
        if self.enable_cpu_kv_offload:
            # 如果 prompt prefix 正在异步写回，抢占时不能把这些 block 立刻放回 free list。
            # 标记为 release_on_complete，让 D2H 完成后再释放。
            self._mark_pending_writeback_release(seq.block_table)
            protected = self._pending_writeback_block_ids()
        if lost_tokens:
            self.metrics["preemption_count"] += 1
            seq.num_preemptions += 1
            if self.enable_cpu_kv_offload:
                # CPU offload 打开时，完整 prefix block 已有 CPU backing 或正在写回。
                # 抢占后真正需要 recompute 的，只是 decode 产生的最后一个未满 block。
                seq.recompute_pending_tokens += self._decode_tail_tokens_without_prefix_backing(seq)
            else:
                # baseline 没有 CPU backing，抢占释放的 cached tokens 都要靠后续 prefill 重算。
                seq.recompute_pending_tokens += lost_tokens
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq, protected)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            prefill_finished = is_prefill and seq.num_cached_tokens == seq.num_tokens
            if prefill_finished:
                if self.enable_lazy_cpu_kv_writeback:
                    # V3：prefill 完不立刻全量备份。此时 request 还要 decode，
                    # prefix blocks 仍是 active；等 finish/preempt 释放到 inactive 后，
                    # schedule() 里的 lazy 窗口才会从 LRU victim 侧选择一部分写回。
                    seq.eager_prefix_writeback_done = True
                else:
                    # V1/V2：prefill 完后立即按 request eager 备份完整 prefix。
                    self._start_eager_prefix_writeback(seq, release_on_complete=False)
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                if self.enable_cpu_kv_offload:
                    # V1/V2 在 prefill_finished 时已经 eager writeback。V3 lazy 模式下，
                    # finish 只把完整 prefix 释放到 inactive LRU；是否写回 CPU 交给下一轮 schedule 统一检查。
                    if not self.enable_lazy_cpu_kv_writeback:
                        # prompt 很短或 finish 时 eager 还没提交过，就补一次；
                        # request 已结束，所以 pending D2H 完成后可以顺手 release。
                        self._start_eager_prefix_writeback(seq, release_on_complete=True)
                    self._mark_pending_writeback_release(seq.block_table)
                protected = self._pending_writeback_block_ids() if self.enable_cpu_kv_offload else set()
                # 已完成写回的 block 可以释放；仍在 D2H 的 block 继续 protected，后续 poll 完成后再释放。
                self.block_manager.deallocate(seq, protected)
                self.running.remove(seq)
