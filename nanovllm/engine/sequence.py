from copy import copy
from enum import Enum, auto
from itertools import count
import numpy as np
import xxhash

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.recompute_pending_tokens = 0
        self.num_preemptions = 0
        # V1/V2 eager offload：每个 request 的完整 prompt prefix 最多提交一次 D2H。
        # V3 lazy 模式会把这个标记置 True，用来明确跳过 eager 路径。
        self.eager_prefix_writeback_done = False
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        # Prefix block contents are immutable once a block is full. Cache the
        # chained hashes on the request so lookup and hash_blocks never hash the
        # same tokens twice.
        self._block_hashes = []

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block_token_ids(self, local_block_idx):
        # prefix cache 用完整 block token_ids 做 hash 校验；单独封装方便 GPU/CPU 两级 cache 共用。
        assert 0 <= local_block_idx < self.num_blocks
        start = local_block_idx * self.block_size
        end = start + self.block_size
        return self.token_ids[start:end]

    @staticmethod
    def compute_hash(token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.asarray(token_ids, dtype=np.int64).tobytes())
        return h.intdigest()

    def block_hash(self, local_block_idx: int):
        assert 0 <= local_block_idx < self.num_blocks
        while len(self._block_hashes) <= local_block_idx:
            idx = len(self._block_hashes)
            token_ids = self.block_token_ids(idx)
            assert len(token_ids) == self.block_size
            prefix = self._block_hashes[-1] if self._block_hashes else -1
            self._block_hashes.append(self.compute_hash(token_ids, prefix))
        return self._block_hashes[local_block_idx]

    def block(self, i):
        return self.block_token_ids(i)

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.recompute_pending_tokens, self.num_preemptions, self.eager_prefix_writeback_done, self.block_table, last_state)

    def __setstate__(self, state):
        if len(state) == 8:
            self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.recompute_pending_tokens, self.num_preemptions, self.block_table, last_state = state
            self.eager_prefix_writeback_done = False
        else:
            self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.recompute_pending_tokens, self.num_preemptions, self.eager_prefix_writeback_done, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
        self._block_hashes = []
