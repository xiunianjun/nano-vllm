from copy import copy
from enum import Enum, auto
from itertools import count

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
        # V1 prefix offload: prompt prefill 完成后只主动写回一次 CPU backing KV。
        # decode 新产生的 tokens 暂时不进入 prefix cache，后续请求重新 prefill 后再纳入。
        self.prefix_writeback_started = False
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

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

    def block(self, i):
        return self.block_token_ids(i)

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.recompute_pending_tokens, self.num_preemptions, self.prefix_writeback_started, self.block_table, last_state)

    def __setstate__(self, state):
        if len(state) == 8:
            self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.recompute_pending_tokens, self.num_preemptions, self.block_table, last_state = state
            self.prefix_writeback_started = False
        else:
            self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.recompute_pending_tokens, self.num_preemptions, self.prefix_writeback_started, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
