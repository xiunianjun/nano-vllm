from collections import deque, OrderedDict

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
        # Keep transfer state on the block instead of duplicating it in a
        # BlockManager set.
        self.writeback_pending = False

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
        self.writeback_pending = False

    def invalidate(self):
        self.hash = -1
        self.token_ids = []
        self.writeback_pending = False


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, enable_prefix_cache: bool = True, enable_lru_retention: bool = True):
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache
        self.enable_lru_retention = enable_lru_retention
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        # CPU prefix cache 这里只保存 hash -> token_ids 的元数据，用来判断 CPU hit；
        # 真正的 CPU KV tensor 存在 ModelRunner.cpu_prefix_cache 里。
        self.cpu_hash_to_token_ids: dict[int, list[int]] = dict()
        # free: 没有任何有效 KV，可直接分配给新 request/prefill 覆盖。
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # active: 正在被 running/waiting request 引用；ref_count > 0 时不能被 LRU 淘汰。
        # inactive LRU: request 已释放引用，但 GPU KV 和 hash 仍有效。
        # OrderedDict 左侧是最久未使用的 victim，右侧是最近释放/命中的 block。
        self.inactive_block_ids: OrderedDict[int, None] = OrderedDict()
        # D2H pending 的 inactive block 不能被覆盖；等 CUDA event 完成后，
        # Scheduler 会 register_cpu_blocks()，它才变成“可安全淘汰”的 CPU-backed victim。
        # O(1) decode admission: the old path scanned the full inactive LRU
        # for every generated token.
        self.evictable_inactive_block_count = 0
        self.reset_metrics()

    def reset_metrics(self):
        self.metrics = {
            "gpu_lru_hit_block_count": 0,
            "gpu_lru_hit_token_count": 0,
            "gpu_lru_eviction_count": 0,
            "gpu_lru_evicted_cpu_backed_block_count": 0,
            "gpu_lru_evicted_gpu_only_block_count": 0,
            "gpu_lru_cached_block_peak": len(self.inactive_block_ids),
            "lazy_writeback_scheduled_block_count": 0,
        }

    def gpu_residency_for_cpu_eviction(self, protected_window_blocks: int):
        """Return GPU-resident hashes and the LRU victim-front hashes to protect.

        CPU backing outside the protected victim window is redundant capacity: the
        corresponding KV is still available on GPU and is not next in line for GPU
        eviction.  The ModelRunner uses these snapshots only as CPU eviction hints;
        BlockManager remains the source of truth for residency and safety.
        """
        gpu_resident_hashes = set(self.hash_to_block_id)
        protected_hashes = set()
        for i, block_id in enumerate(self.inactive_block_ids):
            if i >= protected_window_blocks:
                break
            block = self.blocks[block_id]
            if block.hash != -1:
                protected_hashes.add(block.hash)
        return gpu_resident_hashes, protected_hashes

    def get_metrics(self, protected_window_blocks: int = 0):
        metrics = dict(self.metrics)
        gpu_prefix_hashes = set(self.hash_to_block_id)
        cpu_prefix_hashes = set(self.cpu_hash_to_token_ids)
        duplicate_hashes = gpu_prefix_hashes & cpu_prefix_hashes
        _gpu_resident, protected_hashes = self.gpu_residency_for_cpu_eviction(protected_window_blocks)
        active_hashes = {
            block.hash for block in self.blocks
            if block.ref_count > 0 and block.hash != -1
        }
        metrics["gpu_prefix_cached_block_count"] = len(gpu_prefix_hashes)
        metrics["cpu_gpu_duplicate_block_count"] = len(duplicate_hashes)
        metrics["cpu_gpu_active_duplicate_block_count"] = len(duplicate_hashes & active_hashes)
        metrics["cpu_gpu_protected_duplicate_block_count"] = len(duplicate_hashes & protected_hashes)
        metrics["cpu_gpu_unprotected_duplicate_block_count"] = len(
            duplicate_hashes - active_hashes - protected_hashes
        )
        inactive_cpu_backed = sum(
            1 for block_id in self.inactive_block_ids
            if not self.blocks[block_id].writeback_pending and self._is_cpu_backed(block_id)
        )
        inactive_pending = sum(1 for block_id in self.inactive_block_ids if self.blocks[block_id].writeback_pending)
        metrics["gpu_lru_cached_block_count"] = len(self.inactive_block_ids)
        metrics["inactive_cpu_backed_block_count"] = inactive_cpu_backed
        metrics["inactive_gpu_only_block_count"] = len(self.inactive_block_ids) - inactive_cpu_backed - inactive_pending
        metrics["inactive_pending_writeback_block_count"] = inactive_pending
        metrics["inactive_safe_or_pending_block_count"] = inactive_cpu_backed + inactive_pending
        metrics["safe_allocatable_block_count"] = len(self.free_block_ids) + inactive_cpu_backed
        return metrics

    def free_block_count(self) -> int:
        return len(self.free_block_ids)

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        return Sequence.compute_hash(token_ids, prefix)

    def _remove_hash_mapping(self, block_id: int):
        block = self.blocks[block_id]
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]

    def _available_block_count(self, exclude_block_ids: set[int] | None = None) -> int:
        # 可分配容量 = 真正 free 的 blocks + 可以被淘汰的 inactive LRU blocks。
        # exclude_block_ids 是本次 allocate 中已经作为 GPU hit 复用的 blocks，不能又被选作 victim。
        available = len(self.free_block_ids) + self.evictable_inactive_block_count
        if exclude_block_ids:
            available -= sum(
                block_id in self.inactive_block_ids and not self.blocks[block_id].writeback_pending
                for block_id in exclude_block_ids
            )
        return available

    # 实际抢占逻辑
    def _allocate_block(self, exclude_block_ids: set[int] | None = None) -> int:
        # 分配优先吃 free list；只有 free 不够时，才从 inactive LRU 淘汰一个旧 prefix。
        if self.free_block_ids:
            block_id = self.free_block_ids.popleft()
        else:
            block_id = self._evict_inactive_block(exclude_block_ids)
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # 被重新分配后，这个物理 block 上的旧 hash 不再代表有效 prefix cache。
        self._remove_hash_mapping(block_id)
        block.reset()
        return block_id

    def _deallocate_block(self, block_id: int):
        block = self.blocks[block_id]
        assert block.ref_count == 0
        self._remove_hash_mapping(block_id)
        block.invalidate()
        self.free_block_ids.append(block_id)

    def _deactivate_block(self, block_id: int):
        block = self.blocks[block_id]
        assert block.ref_count == 0 and block.hash != -1
        # V2 的关键点：release request 引用时不清 KV、不删 hash，而是放入 inactive LRU。
        # 后续相同 prefix 再来，可以直接 GPU hit；显存紧张时再按 LRU victim 覆盖。
        self.inactive_block_ids[block_id] = None
        self.inactive_block_ids.move_to_end(block_id)
        if not block.writeback_pending:
            self.evictable_inactive_block_count += 1
        self.metrics["gpu_lru_cached_block_peak"] = max(
            self.metrics["gpu_lru_cached_block_peak"], len(self.inactive_block_ids)
        )

    def _activate_inactive_block(self, block_id: int):
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # inactive block 被 prefix lookup 命中后重新变成 active 引用。
        # 这里不需要任何 H2D copy，因为 KV 一直留在 GPU 上。
        self.inactive_block_ids.pop(block_id)
        if not block.writeback_pending:
            self.evictable_inactive_block_count -= 1
        block.ref_count = 1
        self.metrics["gpu_lru_hit_block_count"] += 1
        self.metrics["gpu_lru_hit_token_count"] += self.block_size

    def _is_cpu_backed(self, block_id: int) -> bool:
        block = self.blocks[block_id]
        return block.hash != -1 and self.has_cpu_block(block.hash, block.token_ids)

    def _is_safe_or_pending_victim(self, block_id: int) -> bool:
        return self.blocks[block_id].writeback_pending or self._is_cpu_backed(block_id)

    def victim_window_safe_or_pending_count(self, target_safe_blocks: int) -> int:
        # V3 只关心 LRU victim 前沿，而不是整个 inactive 队列。
        # 如果热 block 被复用过，它会回到 MRU 端；即使 CPU-backed，也不该拿来填 victim window。
        count = 0
        for i, block_id in enumerate(self.inactive_block_ids):
            if i >= target_safe_blocks:
                break
            if self._is_safe_or_pending_victim(block_id):
                count += 1
        return count

    def select_lazy_writeback_entries(self, target_safe_blocks: int, pending_hashes: set[int]):
        # V3 lazy writeback 只“备份可能马上被淘汰的 inactive blocks”，不碰 active request。
        # target_safe_blocks 对应 LRU victim 前沿窗口：窗口内 safe/pending 不够，才补写回。
        # 这样刚被复用、位于 MRU 端的 CPU-backed block 不会让前沿窗口产生虚假的安全感。
        deficit = target_safe_blocks - self.victim_window_safe_or_pending_count(target_safe_blocks)
        if deficit <= 0:
            return []
        entries = []
        # OrderedDict 左侧是 LRU victim 侧。只从 victim window 内挑 GPU-only blocks 写回；
        # 窗口之外的热端 block 即使没有 CPU backing，也暂时不碰。
        for i, block_id in enumerate(self.inactive_block_ids):
            if i >= target_safe_blocks or len(entries) >= deficit:
                break
            if self.blocks[block_id].writeback_pending:
                continue
            block = self.blocks[block_id]
            # 三类 block 不需要再次写回：无有效 hash、相同 hash 已在 pending、CPU 已有 backing。
            if block.hash == -1 or block.hash in pending_hashes or self.has_cpu_block(block.hash, block.token_ids):
                continue
            entries.append((block.hash, block_id, block.token_ids))
        self.metrics["lazy_writeback_scheduled_block_count"] += len(entries)
        return entries

    def mark_cpu_writeback_pending(self, entries):
        # D2H 提交后立即 protected，防止 allocator 在 copy 完成前覆盖该 GPU block。
        for _h, block_id, _tokens in entries:
            block = self.blocks[block_id]
            if block.writeback_pending:
                continue
            block.writeback_pending = True
            if block_id in self.inactive_block_ids:
                self.evictable_inactive_block_count -= 1

    def _evict_inactive_block(self, exclude_block_ids: set[int] | None = None) -> int:
        exclude_block_ids = exclude_block_ids or set()
        victim = None
        # 真正的 GPU 驱逐发生在 allocation 路径：free list 为空时才调用这里。
        # 不再“全局优先 CPU-backed”，而是严格沿 LRU victim 前沿往后找。
        # 这样刚被复用过、回到 MRU 端的 CPU-backed block 不会被跨顺序提前淘汰。
        for block_id in self.inactive_block_ids:
            # exclude 是本次 allocate 已经命中的 GPU prefix，pending 是 copy stream 正在读取的 block；
            # 两者都不能在同一次 allocation 里被选成 victim。
            if block_id in exclude_block_ids or self.blocks[block_id].writeback_pending:
                continue
            victim = block_id
            break
        assert victim is not None
        was_cpu_backed = self._is_cpu_backed(victim)
        self.inactive_block_ids.pop(victim)
        self.evictable_inactive_block_count -= 1
        # victim 被覆盖前必须删掉 GPU hash mapping，否则后续会误判成 GPU hit。
        self._remove_hash_mapping(victim)
        self.blocks[victim].invalidate()
        self.metrics["gpu_lru_eviction_count"] += 1
        if was_cpu_backed:
            self.metrics["gpu_lru_evicted_cpu_backed_block_count"] += 1
        else:
            self.metrics["gpu_lru_evicted_gpu_only_block_count"] += 1
        return victim

    # 返回 request 的 block 布局：[GPU, CPU, miss]
    def get_allocate_plan(self, seq: Sequence, enable_cpu_cache: bool = False):
        # 为新 request 查找最长连续 prefix：优先 GPU hit，其次 CPU hit，遇到 miss 就停止。
        # prefix cache 必须从 prompt 开头连续命中，不能跳过中间 block。
        if not self.enable_prefix_cache:
            if self._available_block_count() < seq.num_blocks:
                return None
            return {"sources": []}

        sources = []
        gpu_source_block_ids = set()
        free_needed = seq.num_blocks
        for local_block_idx in range(seq.num_blocks - 1):
            block_token_ids = seq.block_token_ids(local_block_idx)
            h = seq.block_hash(local_block_idx)
            global_block_id = self.hash_to_block_id.get(h, -1)

            if global_block_id != -1 and self.blocks[global_block_id].token_ids == block_token_ids:
                sources.append(("gpu", h, block_token_ids))
                gpu_source_block_ids.add(global_block_id)
                free_needed -= 1
                continue
            if enable_cpu_cache and self.cpu_hash_to_token_ids.get(h) == block_token_ids:
                sources.append(("cpu", h, block_token_ids))
                continue
            break

        if self._available_block_count(gpu_source_block_ids) < free_needed:
            return None
        return {"sources": sources}

    def can_allocate(self, seq: Sequence) -> int:
        plan = self.get_allocate_plan(seq, False)
        return len(plan["sources"]) if plan is not None else -1

    # allocate 根据 plan 同时处理三类 block：
    # GPU hit 直接引用；CPU hit 分配 GPU block 并返回 restore_entries；miss 分配空 block 给 prefill 写入。
    def allocate(self, seq: Sequence, num_cached_blocks_or_plan):
        assert not seq.block_table
        if isinstance(num_cached_blocks_or_plan, int):
            plan = self.get_allocate_plan(seq, False)
            assert plan is not None and len(plan["sources"]) == num_cached_blocks_or_plan
        else:
            plan = num_cached_blocks_or_plan

        restore_entries = []
        # 同一次 allocate 里，已经 GPU hit 的 blocks 会被 seq 引用。
        # 如果后面还需要给 CPU hit/miss 分配新 block，不能把这些刚命中的 inactive blocks 淘汰掉。
        gpu_source_block_ids = {
            self.hash_to_block_id[h]
            for source, h, block_token_ids in plan["sources"]
            if source == "gpu"
        }
        for source, h, block_token_ids in plan["sources"]:
            if source == "gpu":
                global_block_id = self.hash_to_block_id[h]
                block = self.blocks[global_block_id]
                if global_block_id in self.inactive_block_ids:
                    # V2 GPU LRU hit：inactive -> active，只改元数据，不做 copy。
                    self._activate_inactive_block(global_block_id)
                else:
                    # 兼容原 prefix cache：block 可能已经被其他 active request 共享引用。
                    block.ref_count += 1
                seq.block_table.append(global_block_id)
            else:
                # CPU hit：先分配一个 GPU block，占好 hash 映射，再由 Scheduler 调 ModelRunner 做同步 H2D。
                global_block_id = self._allocate_block(gpu_source_block_ids)
                self.blocks[global_block_id].update(h, block_token_ids)
                self.hash_to_block_id[h] = global_block_id
                restore_entries.append((h, global_block_id))
                seq.block_table.append(global_block_id)

        for _ in range(len(plan["sources"]), seq.num_blocks):
            seq.block_table.append(self._allocate_block(gpu_source_block_ids))
        seq.num_cached_tokens = len(plan["sources"]) * self.block_size
        return restore_entries

    def prefix_entries(self, seq: Sequence):
        entries = []
        for global_block_id in seq.block_table:
            block = self.blocks[global_block_id]
            if block.hash == -1:
                break
            entries.append((block.hash, global_block_id, block.token_ids))
        return entries

    # 去重，CPU 已有不再重复写回
    def has_cpu_block(self, h: int, block_token_ids: list[int]) -> bool:
        return self.cpu_hash_to_token_ids.get(h) == block_token_ids

    def register_cpu_blocks(self, entries):
        # ModelRunner 确认 D2H 完成后调用：metadata 与真实 CPU tensor 同步可见。
        # discard pending 后，该 inactive block 就能作为 CPU-backed victim 被优先淘汰。
        for h, global_block_id, block_token_ids in entries:
            block = self.blocks[global_block_id]
            if block.writeback_pending:
                block.writeback_pending = False
                if global_block_id in self.inactive_block_ids:
                    self.evictable_inactive_block_count += 1
            self.cpu_hash_to_token_ids[h] = block_token_ids

    def unregister_cpu_blocks(self, hashes):
        # CPU cache 容量限制可能会淘汰 backing tensor；Scheduler 收到后删掉 metadata，
        # 防止后续把已经被清掉的 prefix 误判为 CPU hit。
        for h in hashes:
            self.cpu_hash_to_token_ids.pop(h, None)

    def release_blocks(self, global_block_ids):
        # global_block_ids 按 request 内的 prefix 顺序排列：越靠前的 block 代表越短、越通用的 prefix。
        # 这里按逆序释放，和 vLLM 的 prefix-aware LRU/free 策略一致：tail block 先进入队列左侧，
        # 在相同 recency 下更早被重新分配/淘汰；靠近 prefix 根部的 block 后进入队列，更容易留久一点。
        for global_block_id in reversed(global_block_ids):
            block = self.blocks[global_block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                if self.enable_prefix_cache and self.enable_lru_retention and block.hash != -1:
                    # V2：完整 prefix block 释放后进入 inactive LRU，保留 GPU KV 供后续复用。
                    self._deactivate_block(global_block_id)
                else:
                    # V1 或非 prefix block：直接释放，后续复用只能依赖 CPU backing 或 recompute。
                    self._deallocate_block(global_block_id)

    def deallocate(self, seq: Sequence, skip_block_ids: set[int] | None = None):
        skip_block_ids = skip_block_ids or set()
        releasable = [block_id for block_id in seq.block_table if block_id not in skip_block_ids]
        self.release_blocks(releasable)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return self._available_block_count() >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence) -> bool:
        if len(seq) % self.block_size == 1:
            # decode 跨 block 边界时才需要新 block；新 decode block 暂不算 prefix cache。
            seq.block_table.append(self._allocate_block())
            return True
        return False

    def hash_blocks(self, seq: Sequence):
        if not self.enable_prefix_cache:
            return
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end:
            return
        for local_block_idx in range(start, end):
            global_block_id = seq.block_table[local_block_idx]
            block = self.blocks[global_block_id]
            block_token_ids = seq.block_token_ids(local_block_idx)
            # block 只有写满后才进入 prefix cache。hash 串上前一个 block hash，保证只能连续 prefix 命中。
            h = seq.block_hash(local_block_idx)
            block.update(h, block_token_ids)
            self.hash_to_block_id[h] = global_block_id
