from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, enable_prefix_cache: bool = True):
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        # CPU prefix cache 这里只保存 hash -> token_ids 的元数据，用来判断 CPU hit；
        # 真正的 CPU KV tensor 存在 ModelRunner.cpu_prefix_cache 里。
        self.cpu_hash_to_token_ids: dict[int, list[int]] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # 返回 request 的 block 布局：[GPU, CPU, miss]
    def get_allocate_plan(self, seq: Sequence, enable_cpu_cache: bool = False):
        # 为新 request 查找最长连续 prefix：优先 GPU hit，其次 CPU hit，遇到 miss 就停止。
        # prefix cache 必须从 prompt 开头连续命中，不能跳过中间 block。
        if not self.enable_prefix_cache:
            if len(self.free_block_ids) < seq.num_blocks:
                return None
            return {"sources": []}

        h = -1
        sources = []
        free_needed = seq.num_blocks
        for local_block_idx in range(seq.num_blocks - 1):
            block_token_ids = seq.block_token_ids(local_block_idx)
            # Chain the hash so a block only matches when the full prefix matches.
            h = self.compute_hash(block_token_ids, h)
            global_block_id = self.hash_to_block_id.get(h, -1)

            if global_block_id != -1 and self.blocks[global_block_id].token_ids == block_token_ids:
                sources.append(("gpu", h, block_token_ids))
                if global_block_id in self.used_block_ids:
                    free_needed -= 1
                continue
            if enable_cpu_cache and self.cpu_hash_to_token_ids.get(h) == block_token_ids:
                # CPU hit 仍然需要一个新的 GPU block 作为 restore 目的地。
                sources.append(("cpu", h, block_token_ids))
                # CPU 上的 block 也需要分配 GPU block 来存储，不修改 free_needed
                continue
            break

        if len(self.free_block_ids) < free_needed:
            # V1 当前不在这里主动驱逐；空间不够时交给 Scheduler 等 pending writeback 或暂停调度。
            # TODO(V2): 这里可以接入 OPT/LRU victim 选择，优先驱逐已写回 CPU 的 GPU blocks。
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
        for source, h, block_token_ids in plan["sources"]:
            if source == "gpu":
                # GPU resident prefix：直接复用原 block，不产生 H2D。
                global_block_id = self.hash_to_block_id[h]
                block = self.blocks[global_block_id]
                if global_block_id in self.used_block_ids:
                    block.ref_count += 1
                else:
                    block.ref_count = 1
                    self.free_block_ids.remove(global_block_id)
                    self.used_block_ids.add(global_block_id)
            else:
                # CPU resident prefix：先占一个 GPU block 并登记 hash，随后由 ModelRunner restore KV 内容。
                global_block_id = self._allocate_block()
                self.blocks[global_block_id].update(h, block_token_ids)
                self.hash_to_block_id[h] = global_block_id
                restore_entries.append((h, global_block_id))
            seq.block_table.append(global_block_id)

        for _ in range(len(plan["sources"]), seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = len(plan["sources"]) * self.block_size
        return restore_entries

    def prefix_entries(self, seq: Sequence):
        # request 完成时只写回已经形成完整 block hash 的 prefix；
        # 末尾未满 block 不写回，因为后续 prefix lookup 也是按完整 block 命中。
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
        # async D2H 完成后登记 CPU-resident 元数据；之后相同 prefix 可走 CPU hit。
        for h, _global_block_id, block_token_ids in entries:
            self.cpu_hash_to_token_ids[h] = block_token_ids

    def release_blocks(self, global_block_ids):
        # 主动写回完成或 request 结束时释放 GPU block 引用计数。
        for global_block_id in reversed(global_block_ids):
            block = self.blocks[global_block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(global_block_id)

    def deallocate(self, seq: Sequence, skip_block_ids: set[int] | None = None):
        # skip_block_ids 是异步 D2H writeback 尚未完成的 protected blocks；
        # 它们不能回到 free list，否则可能在 copy 过程中被新 request 覆盖。
        skip_block_ids = skip_block_ids or set()
        releasable = [block_id for block_id in seq.block_table if block_id not in skip_block_ids]
        self.release_blocks(releasable)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        if not self.enable_prefix_cache:
            return
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        # prefill 完成后更新 GPU prefix hash 表；CPU backing 只在 request 完成后的 writeback 阶段产生。
        for local_block_idx in range(start, end):
            global_block_id = seq.block_table[local_block_idx]
            block = self.blocks[global_block_id]
            block_token_ids = seq.block_token_ids(local_block_idx)
            h = self.compute_hash(block_token_ids, h)
            block.update(h, block_token_ids)
            self.hash_to_block_id[h] = global_block_id
