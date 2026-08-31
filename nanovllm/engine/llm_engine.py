# generate -> scheduler
#             model_runner

import atexit
import statistics
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        if config.enable_cpu_kv_offload:
            # Scheduler 只负责调度决策；真正搬 prefix KV tensor 的 D2H/H2D 操作必须在 ModelRunner 里做，
            # 因为 kv_cache tensor、copy stream 和 CUDA event 都属于 model runner 进程/设备上下文。
            self.scheduler.writeback_prefix_blocks = lambda entries: self.model_runner.call("writeback_prefix_blocks", entries)
            self.scheduler.restore_prefix_blocks = lambda entries: self.model_runner.call("restore_prefix_blocks", entries)
            self.scheduler.poll_prefix_writebacks = lambda wait=False: self.model_runner.call("poll_prefix_writebacks", wait)
        self.request_start_times = {}
        self.request_latencies = []
        self.queueing_latencies = []
        self.first_scheduled_recorded = set()
        self.step_metrics = self._new_step_metrics()
        self.first_token_recorded = set()
        atexit.register(self.exit)

    def _new_step_metrics(self):
        return {
            "prefill_step_count": 0,
            "prefill_step_time_sec": 0.0,
            "prefill_token_count_timed": 0,
            "decode_step_count": 0,
            "decode_step_time_sec": 0.0,
            "decode_token_count_timed": 0,
            "schedule_time_sec": 0.0,
            "model_runner_call_time_sec": 0.0,
            "postprocess_time_sec": 0.0,
            "ttft_latencies": [],
        }

    def exit(self):
        if not hasattr(self, "model_runner"):
            return
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq

    def reset_metrics(self):
        self.scheduler.reset_metrics()
        self.model_runner.call("reset_prefix_transfer_metrics")
        self.request_start_times.clear()
        self.request_latencies.clear()
        self.queueing_latencies.clear()
        self.first_scheduled_recorded.clear()
        self.step_metrics = self._new_step_metrics()
        self.first_token_recorded = set()

    def get_metrics(self):
        metrics = self.scheduler.get_metrics()
        metrics.update(self.model_runner.get_prefix_transfer_metrics())
        latencies = self.request_latencies
        ttft_latencies = self.step_metrics["ttft_latencies"]
        queueing_latencies = self.queueing_latencies
        prefill_step_count = self.step_metrics["prefill_step_count"]
        decode_step_count = self.step_metrics["decode_step_count"]
        prefill_step_time = self.step_metrics["prefill_step_time_sec"]
        decode_step_time = self.step_metrics["decode_step_time_sec"]
        metrics.update({
            "request_latency_count": len(latencies),
            "request_latency_avg": sum(latencies) / len(latencies) if latencies else 0.0,
            "request_latency_median": statistics.median(latencies) if latencies else 0.0,
            "request_latency_max": max(latencies) if latencies else 0.0,
            "request_latency_min": min(latencies) if latencies else 0.0,
            "ttft_latency_count": len(ttft_latencies),
            "ttft_latency_avg": sum(ttft_latencies) / len(ttft_latencies) if ttft_latencies else 0.0,
            "ttft_latency_median": statistics.median(ttft_latencies) if ttft_latencies else 0.0,
            "ttft_latency_max": max(ttft_latencies) if ttft_latencies else 0.0,
            "ttft_latency_min": min(ttft_latencies) if ttft_latencies else 0.0,
            "queueing_latency_count": len(queueing_latencies),
            "queueing_latency_avg": sum(queueing_latencies) / len(queueing_latencies) if queueing_latencies else 0.0,
            "queueing_latency_median": statistics.median(queueing_latencies) if queueing_latencies else 0.0,
            "queueing_latency_max": max(queueing_latencies) if queueing_latencies else 0.0,
            "queueing_latency_min": min(queueing_latencies) if queueing_latencies else 0.0,
            "prefill_step_count": prefill_step_count,
            "prefill_step_time_sec": prefill_step_time,
            "prefill_step_time_total_sec": prefill_step_time,
            "prefill_step_time_avg_sec": (
                prefill_step_time / prefill_step_count if prefill_step_count else 0.0
            ),
            "prefill_timed_tokens": self.step_metrics["prefill_token_count_timed"],
            "prefill_timed_tok_per_sec": (
                self.step_metrics["prefill_token_count_timed"] / prefill_step_time
                if prefill_step_time else 0.0
            ),
            "decode_step_count": decode_step_count,
            "decode_step_time_sec": decode_step_time,
            "decode_step_time_total_sec": decode_step_time,
            "decode_step_time_avg_sec": (
                decode_step_time / decode_step_count if decode_step_count else 0.0
            ),
            "decode_timed_tokens": self.step_metrics["decode_token_count_timed"],
            "schedule_time_sec": self.step_metrics["schedule_time_sec"],
            "model_runner_call_time_sec": self.step_metrics["model_runner_call_time_sec"],
            "postprocess_time_sec": self.step_metrics["postprocess_time_sec"],
        })
        return metrics

    def step(self):
        schedule_start = perf_counter()
        seqs, is_prefill = self.scheduler.schedule()
        self.step_metrics["schedule_time_sec"] += perf_counter() - schedule_start
        if is_prefill:
            for seq in seqs:
                if seq.seq_id not in self.first_scheduled_recorded:
                    self.queueing_latencies.append(schedule_start - self.request_start_times[seq.seq_id])
                    self.first_scheduled_recorded.add(seq.seq_id)
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        t = perf_counter()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.step_metrics["model_runner_call_time_sec"] += perf_counter() - t
        t = perf_counter()
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        self.step_metrics["postprocess_time_sec"] += perf_counter() - t
        if not is_prefill:
            now = perf_counter()
            for seq in seqs:
                if seq.seq_id not in self.first_token_recorded and seq.num_completion_tokens >= 1:
                    self.step_metrics["ttft_latencies"].append(now - self.request_start_times[seq.seq_id])
                    self.first_token_recorded.add(seq.seq_id)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        start_time = perf_counter()
        for prompt, sp in zip(prompts, sampling_params):
            seq = self.add_request(prompt, sp)
            self.request_start_times[seq.seq_id] = start_time
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        self.first_token_recorded = set()
        while not self.is_finished():
            t = perf_counter()
            # for each step():
            #     scheduler.schedule()  →  prefill OR decode
            #     model_runner.forward()  →  run model
            output, num_tokens = self.step()
            step_time = perf_counter() - t
            if num_tokens > 0:
                self.step_metrics["prefill_step_count"] += 1
                self.step_metrics["prefill_step_time_sec"] += step_time
                self.step_metrics["prefill_token_count_timed"] += num_tokens
            else:
                self.step_metrics["decode_step_count"] += 1
                self.step_metrics["decode_step_time_sec"] += step_time
                self.step_metrics["decode_token_count_timed"] += -num_tokens
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / step_time
                else:
                    decode_throughput = -num_tokens / step_time
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            # seq_id: 请求编号；token_id: 词表编号
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids # 拼接后结果
                self.request_latencies.append(perf_counter() - self.request_start_times.pop(seq_id, t))
                if use_tqdm:
                    pbar.update(1)
        pbar.close()
        # 按请求顺序返回结果；转化token id为文本
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
