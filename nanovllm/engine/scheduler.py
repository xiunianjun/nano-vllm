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
        # V1 prefix offload 通过 LLMEngine 注入 ModelRunner 侧的 copy 回调。
        self.writeback_prefix_blocks = None
        self.restore_prefix_blocks = None
        self.poll_prefix_writebacks = None
        # key 是异步 D2H writeback id，value 是本次写回涉及的 prefix blocks；
        # pending 期间这些 GPU blocks 仍保存有效 KV，但不能回到 free list。
        self.pending_prefix_writebacks = {}
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
            # CPU prefix cache 指标用于区分：GPU 直接命中 vs CPU 命中后 restore。
            "cpu_prefix_cache_hit_count": 0,
            "cpu_prefix_cache_restored_token_count": 0,
            "pending_prefix_writeback_count": 0,
        }

    def get_metrics(self):
        return {
            "preemption_count": self.metrics["preemption_count"],
            "recomputed_token_count": self.metrics["recomputed_token_count"],
            "prefill_token_count": self.metrics["prefill_token_count"],
            "prefix_cache_lookup_count": self.metrics["prefix_cache_lookup_count"],
            "prefix_cache_reused_token_count": self.metrics["prefix_cache_reused_token_count"],
            "cpu_prefix_cache_hit_count": self.metrics["cpu_prefix_cache_hit_count"],
            "cpu_prefix_cache_restored_token_count": self.metrics["cpu_prefix_cache_restored_token_count"],
            # 返回实时 pending 数，避免 reset 后 metrics 里的旧值和实际状态不一致。
            "pending_prefix_writeback_count": len(self.pending_prefix_writebacks),
        }


    def _pending_writeback_block_ids(self) -> set[int]:
        return {
            block_id
            for pending in self.pending_prefix_writebacks.values()
            for _h, block_id, _tokens in pending["entries"]
        }

    def _pending_writeback_hashes(self) -> set[int]:
        return {
            h
            for pending in self.pending_prefix_writebacks.values()
            for h, _block_id, _tokens in pending["entries"]
        }

    def _mark_pending_writeback_release(self, block_ids: list[int]):
        block_id_set = set(block_ids)
        for pending in self.pending_prefix_writebacks.values():
            if any(block_id in block_id_set for _h, block_id, _tokens in pending["entries"]):
                pending["release_on_complete"] = True

    def _decode_tail_tokens_without_prefix_backing(self, seq: Sequence) -> int:
        # V1 invariant: prompt prefill 完成后，完整 prefix blocks 已经 CPU_RESIDENT 或 WRITEBACK_PENDING。
        # decode 阶段新产生的 KV 暂不进入 prefix cache，因此抢占时只需要重算最后一个未满 block。
        return seq.num_cached_tokens % self.block_size

    def _start_prefix_writeback(self, seq: Sequence, release_on_complete: bool):
        if not self.enable_cpu_kv_offload or seq.prefix_writeback_started:
            return
        assert self.writeback_prefix_blocks is not None
        entries = self.block_manager.prefix_entries(seq)
        pending_hashes = self._pending_writeback_hashes()
        entries = [
            (h, block_id, tokens)
            for h, block_id, tokens in entries
            if h not in pending_hashes and not self.block_manager.has_cpu_block(h, tokens)
        ]
        if not entries:
            seq.prefix_writeback_started = True
            return
        # V1 主动写回：prompt prefill 一完成，就把完整 prefix blocks 异步 D2H 到 CPU。
        # decode 新产生的 tokens 暂时不纳入 prefix cache；后续请求重新 prefill 后再进入 cache。
        # 写回前会去重：CPU 已有或正在 pending writeback 的 prefix block 不再重复 D2H。
        writeback_id = self.writeback_prefix_blocks([(h, block_id) for h, block_id, _tokens in entries])
        if writeback_id is not None:
            self.pending_prefix_writebacks[writeback_id] = {
                "entries": entries,
                "release_on_complete": release_on_complete,
            }
            seq.prefix_writeback_started = True
            self.metrics["pending_prefix_writeback_count"] = len(self.pending_prefix_writebacks)

    # 回收一下写完的 prefix blocks，用于后续使用。wait=True 时阻塞等待
    def _poll_prefix_writebacks(self, wait: bool = False):
        if not self.enable_cpu_kv_offload or not self.pending_prefix_writebacks:
            return
        assert self.poll_prefix_writebacks is not None
        completed_ids = self.poll_prefix_writebacks(wait)
        for writeback_id in completed_ids:
            pending = self.pending_prefix_writebacks.pop(writeback_id)
            entries = pending["entries"]
            # D2H 完成后，才把 prefix 标记为 CPU-resident。
            # 如果 request 仍在 decode，GPU block 继续归该 request 使用；
            # 如果 request 已 finish/preempt 并跳过释放，则这里再真正 release。
            self.block_manager.register_cpu_blocks(entries)
            if pending["release_on_complete"]:
                self.block_manager.release_blocks([block_id for _h, block_id, _tokens in entries])
        self.metrics["pending_prefix_writeback_count"] = len(self.pending_prefix_writebacks)

    def is_finished(self):
        if not self.waiting and not self.running:
            # 所有 request 都结束后，等待最后一批主动写回完成，保证 CPU cache 元数据落稳。
            self._poll_prefix_writebacks(wait=True)
            return True
        self._poll_prefix_writebacks()
        return False

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # 每轮调度前先非阻塞收割已完成的主动写回，释放可复用 GPU blocks。
        self._poll_prefix_writebacks()
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            restore_entries = []
            if not seq.block_table:
                # 读入 request 时按最长连续 prefix 制定分配计划：
                # GPU hit 直接复用 block；GPU miss + CPU hit 先分配 GPU block，
                # 再同步 restore；GPU/CPU 都 miss 的尾部正常 prefill。
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
                # GPU hit 和 CPU hit 都可以跳过对应 prefix prefill；区别是 CPU hit 需要先 H2D restore。
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
                self.metrics["prefix_cache_lookup_count"] += 1
                self.metrics["prefix_cache_reused_token_count"] += gpu_hits * self.block_size
                self.metrics["cpu_prefix_cache_hit_count"] += cpu_hits
                self.metrics["cpu_prefix_cache_restored_token_count"] += cpu_hits * self.block_size
            else:
                plan = None
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                restore_entries = self.block_manager.allocate(seq, plan)    # GPU hit 直接引用，CPU hit 先分配 GPU block，miss/全新请求分配空 block
                if restore_entries:
                    assert self.restore_prefix_blocks is not None
                    self.restore_prefix_blocks(restore_entries) # 同步 H2D 后再继续 prefill 剩余 suffix。
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
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))   # 重新插入队头，保证纯粹的FCFS
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
                seq.recompute_pending_tokens += self._decode_tail_tokens_without_prefix_backing(seq)
            else:
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
                self._start_prefix_writeback(seq, release_on_complete=False)
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                if self.enable_cpu_kv_offload:
                    # 正常情况下 prefix 已在 prefill_finished 时启动 writeback。
                    # 如果 prompt 太短或早停导致还没启动，则在 finish 时补一次，并让完成后释放 block。
                    self._start_prefix_writeback(seq, release_on_complete=True)
                    self._mark_pending_writeback_release(seq.block_table)
                protected = self._pending_writeback_block_ids() if self.enable_cpu_kv_offload else set()
                # 已完成写回的 block 可以释放；仍在 D2H 的 block 继续 protected，后续 poll 完成后再释放。
                self.block_manager.deallocate(seq, protected)
                self.running.remove(seq)


# TODO(V2): 调度感知预取可以放在 Scheduler.schedule() 的 prefill/decode 决策之后。
# 目标是 inspect waiting/running 队列，优先 prefetch 即将运行的 CPU-resident prefix；
# 如果 GPU KV 空间不足，victim 可以用 OPT-style 策略选择当前窗口内最晚再访问的 prefix。
