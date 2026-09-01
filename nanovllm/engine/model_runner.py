import pickle
from collections import OrderedDict
from time import perf_counter
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        # tensor parallel 的 GPU 数量
        # ＝ 模型 shard 的数量
        # ＝ 用多少卡并行跑模型前向
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        # V1 prefix offload: 以 prefix block hash 为 key 保存 CPU backing KV。
        # request 完成后 GPU KV 不立刻清空；只要 block 尚未被覆盖，GPU/CPU 可以同时持有同一份 prefix。
        self.cpu_prefix_cache = OrderedDict()
        # V3 可选 CPU cache cap。超过后按 CPU LRU 淘汰 backing tensor；
        # 被淘汰的 hash 会返回给 Scheduler/BlockManager 删除 CPU-hit metadata。
        self.cpu_prefix_cache_limit_bytes = int(config.cpu_prefix_cache_gb_limit * (1 << 30))
        # 异步 writeback 的完成状态由 CUDA event 标记，Scheduler 轮询完成后再释放 protected GPU block。
        self.pending_prefix_writebacks = []
        self.next_prefix_writeback_id = 0
        self.reset_prefix_transfer_metrics()

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        # 独立 copy stream 用于 D2H/H2D，给后续 compute/copy overlap 留出空间。
        self.copy_stream = torch.cuda.Stream()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            # worker 进程循环逻辑
            # 都会进入到同一个function
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        # 这里有减少序列化的需求吗？好像传递的是seq（似乎包含prompt），所以是有可能有需要的
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:    # 通知所有。event 是每个 worker 的信号量
            event.set()             # 所以说此处一般有负载均衡的逻辑

    def call(self, method_name, *args): # "run", seqs, is_prefill
        # Rank=分布式计算系统中的并行 worker ID
        # worker 可以是：
        #     一张 GPU
        #     一部分 GPU（例如 MIG slice）
        #     整个 CPU 进程
        #     整个节点的一个 worker
        #     一个 TPU core
        #     或者一个由多个 GPU 组成的进程组
        # rank 0 一般是主进程，其他都是worker进程
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)  # 多卡情况下，主进程写入共享内存支持任务RPC
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):    # 初始化 kvcache 相关参数，在显存开辟一个空间
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # 分在多个 GPU 上
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        # 每个head只关注每个token的一部分，所以把所有head的维度加起来就是token总维度
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        if config.num_kvcache_blocks <= 0:
            config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1


    def reset_prefix_transfer_metrics(self):
        cpu_kv_bytes = sum(entry["bytes"] for entry in self.cpu_prefix_cache.values())
        self.prefix_transfer_metrics = {
            # prefix-cache 主线指标：writeback 是完成请求后的主动 D2H，restore 是 CPU hit 后的同步 H2D。
            "cpu_prefix_writeback_count": 0,
            "cpu_prefix_restore_count": 0,
            "cpu_prefix_d2h_bytes": 0,
            "cpu_prefix_h2d_bytes": 0,
            "cpu_prefix_kv_bytes": cpu_kv_bytes,
            "cpu_prefix_kv_bytes_peak": cpu_kv_bytes,
            "cpu_prefix_cache_eviction_count": 0,
            "cpu_prefix_cache_evicted_bytes": 0,
            "cpu_prefix_writeback_latency_sum": 0.0,
            "cpu_prefix_restore_latency_sum": 0.0,
            "cpu_prefix_writeback_latency_max": 0.0,
            "cpu_prefix_restore_latency_max": 0.0,
            "prefill_prepare_wall_sec": 0.0,
            "prefill_sample_prepare_wall_sec": 0.0,
            "prefill_model_cuda_sec": 0.0,
            "prefill_sampler_wall_sec": 0.0,
            "prefill_model_runner_wall_sec": 0.0,
            "prefill_model_runner_count": 0,
        }

    def get_prefix_transfer_metrics(self):
        metrics = dict(self.prefix_transfer_metrics)
        current_cpu_kv_bytes = sum(entry["bytes"] for entry in self.cpu_prefix_cache.values())
        metrics["cpu_prefix_kv_bytes"] = current_cpu_kv_bytes
        metrics["cpu_prefix_kv_bytes_peak"] = max(metrics["cpu_prefix_kv_bytes_peak"], current_cpu_kv_bytes)
        metrics["cpu_prefix_kv_gb"] = current_cpu_kv_bytes / (1024 ** 3)
        metrics["cpu_prefix_kv_gb_peak"] = metrics["cpu_prefix_kv_bytes_peak"] / (1024 ** 3)
        metrics["cpu_prefix_cache_block_count"] = len(self.cpu_prefix_cache)
        writeback_count = metrics["cpu_prefix_writeback_count"]
        restore_count = metrics["cpu_prefix_restore_count"]
        # CUDA event 记录的是整批 block copy 的时间，这里按批次求平均。
        metrics["cpu_prefix_writeback_latency_avg"] = metrics["cpu_prefix_writeback_latency_sum"] / writeback_count if writeback_count else 0.0
        metrics["cpu_prefix_restore_latency_avg"] = metrics["cpu_prefix_restore_latency_sum"] / restore_count if restore_count else 0.0
        return metrics

    def _enforce_cpu_prefix_cache_limit(self):
        if self.cpu_prefix_cache_limit_bytes <= 0:
            return []
        evicted_hashes = []
        # cpu_prefix_cache 也是 OrderedDict：restore/writeback 命中会 move_to_end，
        # 因此 last=False 淘汰的是 CPU 侧最久未使用的 backing。
        while self.prefix_transfer_metrics["cpu_prefix_kv_bytes"] > self.cpu_prefix_cache_limit_bytes and self.cpu_prefix_cache:
            h, entry = self.cpu_prefix_cache.popitem(last=False)
            self.prefix_transfer_metrics["cpu_prefix_kv_bytes"] -= entry["bytes"]
            self.prefix_transfer_metrics["cpu_prefix_cache_eviction_count"] += 1
            self.prefix_transfer_metrics["cpu_prefix_cache_evicted_bytes"] += entry["bytes"]
            evicted_hashes.append(h)
        return evicted_hashes

    def _record_completed_prefix_writeback(self, pending):
        # D2H event 已完成后才把 CPU tensor 放入正式 cache；
        # 这样 Scheduler 看到 CPU hit 时，一定能安全地从 cpu_prefix_cache 取到完整 KV。
        # 对 V3 来说，这一步也是“pending victim”变成“CPU-backed victim”的分界点。
        bytes_written = 0
        for h, cpu_block in pending["blocks"]:
            nbytes = cpu_block.numel() * cpu_block.element_size()
            old = self.cpu_prefix_cache.get(h)
            if old is not None:
                # 相同 prefix 被再次写回时覆盖旧 CPU backing，避免 CPU occupancy 重复计数。
                self.prefix_transfer_metrics["cpu_prefix_kv_bytes"] -= old["bytes"]
            self.cpu_prefix_cache[h] = {"block": cpu_block, "bytes": nbytes}
            self.cpu_prefix_cache.move_to_end(h)
            self.prefix_transfer_metrics["cpu_prefix_kv_bytes"] += nbytes
            bytes_written += nbytes

        latency = pending["start_event"].elapsed_time(pending["done_event"]) / 1000
        self.prefix_transfer_metrics["cpu_prefix_writeback_count"] += len(pending["blocks"])
        self.prefix_transfer_metrics["cpu_prefix_d2h_bytes"] += bytes_written
        self.prefix_transfer_metrics["cpu_prefix_writeback_latency_sum"] += latency
        self.prefix_transfer_metrics["cpu_prefix_writeback_latency_max"] = max(self.prefix_transfer_metrics["cpu_prefix_writeback_latency_max"], latency)
        self.prefix_transfer_metrics["cpu_prefix_kv_bytes_peak"] = max(
            self.prefix_transfer_metrics["cpu_prefix_kv_bytes_peak"], self.prefix_transfer_metrics["cpu_prefix_kv_bytes"]
        )
        return self._enforce_cpu_prefix_cache_limit()

    def poll_prefix_writebacks(self, wait: bool = False):
        # CUDA 没有 Python 层自动回调；这里用 done_event.query() 判断 async D2H 是否完成。
        # wait=False 是普通 scheduler tick 的非阻塞收割；wait=True 只在必须释放 protected blocks 时使用。
        if wait:
            for pending in self.pending_prefix_writebacks:
                pending["done_event"].synchronize()

        completed_ids = []
        evicted_hashes = []
        still_pending = []
        for pending in self.pending_prefix_writebacks:
            if pending["done_event"].query():
                evicted_hashes.extend(self._record_completed_prefix_writeback(pending))
                completed_ids.append(pending["id"])
            else:
                still_pending.append(pending)
        self.pending_prefix_writebacks = still_pending
        return {"completed_ids": completed_ids, "evicted_hashes": evicted_hashes}

    def writeback_prefix_blocks(self, entries):
        # D2H 写回统一走这里：V1/V2 传入的是 prefill 完整 prefix，V3 传入的是
        # inactive LRU victim 侧挑出的少量 blocks。真实 copy 是异步的，返回的 id 交给 Scheduler 跟踪。
        # 每个 block 单独记录 event；已完成的 block 可以先进入 CPU-backed LRU，未完成的继续 protected。
        if not entries:
            return []

        compute_done = torch.cuda.Event()
        torch.cuda.current_stream().record_event(compute_done)
        writeback_ids = []
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(compute_done)
            for h, block_id in entries:
                writeback_id = self.next_prefix_writeback_id
                self.next_prefix_writeback_id += 1
                start_event = torch.cuda.Event(enable_timing=True)
                done_event = torch.cuda.Event(enable_timing=True)
                gpu_block = self.kv_cache[:, :, block_id]
                # pinned CPU tensor 才能让 non_blocking D2H 真正异步排到 copy stream 上。
                cpu_block = torch.empty(gpu_block.shape, dtype=gpu_block.dtype, device="cpu", pin_memory=True)
                start_event.record()
                cpu_block.copy_(gpu_block, non_blocking=True)
                done_event.record()
                self.pending_prefix_writebacks.append({
                    "id": writeback_id,
                    "blocks": [(h, cpu_block)],
                    "start_event": start_event,
                    "done_event": done_event,
                })
                writeback_ids.append(writeback_id)
        return writeback_ids

    def _copy_cpu_blocks_to_gpu(self, block_copies):
        # block_copies 是 (CPU KV block, 新 GPU block id)，目前用于 CPU prefix restore。
        bytes_read = 0
        for cpu_block, block_id in block_copies:
            self.kv_cache[:, :, block_id].copy_(cpu_block, non_blocking=True)
            bytes_read += cpu_block.numel() * cpu_block.element_size()
        return bytes_read

    def restore_prefix_blocks(self, entries):
        # 读入 request：GPU miss + CPU hit 时，同步 H2D restore 到新分配的 GPU block，
        # restore 完成后才能跳过这段 prefix prefill。
        if not entries:
            return
        compute_done = torch.cuda.Event()
        start_event = torch.cuda.Event(enable_timing=True)
        done_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.current_stream().record_event(compute_done)
        # entries 来自 BlockManager.allocate，GPU block 已经占住；这里只负责把 CPU KV 填进去。
        block_copies = []
        for h, block_id in entries:
            # H2D restore 本身也刷新 CPU LRU recency：最近被拿来复用的 prefix 不该先被 CPU cap 淘汰。
            self.cpu_prefix_cache.move_to_end(h)
            block_copies.append((self.cpu_prefix_cache[h]["block"], block_id))
        with torch.cuda.stream(self.copy_stream):
            # H2D 也走 copy stream；V1 这里随后 synchronize，语义上是同步 swapin/restore。
            self.copy_stream.wait_event(compute_done)
            start_event.record()
            bytes_read = self._copy_cpu_blocks_to_gpu(block_copies)
            done_event.record()
        # V1 restore 是同步语义：必须等 H2D 完成，后续 attention 才能安全跳过 restored prefix。
        done_event.synchronize()
        latency = start_event.elapsed_time(done_event) / 1000
        self.prefix_transfer_metrics["cpu_prefix_restore_count"] += len(entries)
        self.prefix_transfer_metrics["cpu_prefix_h2d_bytes"] += bytes_read
        self.prefix_transfer_metrics["cpu_prefix_restore_latency_sum"] += latency
        self.prefix_transfer_metrics["cpu_prefix_restore_latency_max"] = max(self.prefix_transfer_metrics["cpu_prefix_restore_latency_max"], latency)

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            # eg. block_table = [5, 12, 18]. sequence 的 KV Cache 块分布在显卡上的 block 5,12,18中。一个block有512个kvcache slot（对应一个token）。只存旧kvcache block
            # slot_mapping: 地址映射，新的token具体对应哪个kvcache slot。只存新kvcache位置
            # 第一次跑 prefill，此时没有老的block → 直接跳过
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        # CPU->GPU 数据搬运
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:    # 每次只生成一个 token
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # prefill / 强制 eager / 大 batch → 走普通 PyTorch 前向
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # Graph Replay = 不再逐层执行，而是重放一段 GPU 执行指令序列。相当于把多个kernel launch合成一次replay
            # 在capture_cudagraph初始化图
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # LLM 的每次 forward 都是 batch。
        # prefill 把多个请求的 prompt 合并成 mega-batch，
        # decode 把多个请求下一 token 合并成 tiny-batch。
        # 所以说
        # 跨请求 KV Cache 复用是天然成立的，因为本来就是很多个 request 一起跑一起在显存里
        run_start = perf_counter()
        t = perf_counter()
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        prepare_wall = perf_counter() - t
        t = perf_counter()
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        sample_prepare_wall = perf_counter() - t
        start_event = done_event = None
        if is_prefill:
            start_event = torch.cuda.Event(enable_timing=True)
            done_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        logits = self.run_model(input_ids, positions, is_prefill)
        if is_prefill:
            done_event.record()
        # all-reduce之后的next token 候选表，放在rank0上
        # 所以由rank0进行采样：根据概率分布选词
        t = perf_counter()
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        sampler_wall = perf_counter() - t
        if is_prefill:
            done_event.synchronize()
            self.prefix_transfer_metrics["prefill_prepare_wall_sec"] += prepare_wall
            self.prefix_transfer_metrics["prefill_sample_prepare_wall_sec"] += sample_prepare_wall
            self.prefix_transfer_metrics["prefill_model_cuda_sec"] += start_event.elapsed_time(done_event) / 1000
            self.prefix_transfer_metrics["prefill_sampler_wall_sec"] += sampler_wall
            self.prefix_transfer_metrics["prefill_model_runner_wall_sec"] += perf_counter() - run_start
            self.prefix_transfer_metrics["prefill_model_runner_count"] += 1
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
