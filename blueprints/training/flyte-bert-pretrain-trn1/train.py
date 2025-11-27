import glob
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import flyte
import torch
from flyte.io import Dir
from flyteplugins.pytorch.task import Elastic

data_prep_env = flyte.TaskEnvironment(
    name="bert-trainium-data-prep",
    image=flyte.Image.from_debian_base(
        name="bert-trainium-data-prep"
    ).with_pip_packages(
        "datasets==4.4.1",
        "transformers==4.57.1",
        "tokenizers==0.22.1",
        "flyteplugins-pytorch==2.0.0b29",
    ),
    resources=flyte.Resources(cpu=3, memory="20Gi"),
    cache="auto",
)


trainium_env = flyte.TaskEnvironment(
    name="bert-trainium-training",
    image=flyte.Image.from_base(
        image_uri="public.ecr.aws/neuron/pytorch-training-neuronx:2.8.0-neuronx-py311-sdk2.26.0-ubuntu22.04"
    )
    .clone(name="bert-trainium-training")
    .with_env_vars({"UV_PYTHON": "/usr/local/bin/python3.11"})
    .with_apt_packages("git")
    .with_pip_packages(
        "git+https://github.com/flyteorg/flyte-sdk@a70370bbe348d52351beb6c2f4684efa5f387d46",
        "flyteplugins-pytorch==2.0.0b29",
        "transformers==4.57.1",
        "datasets==4.4.1",
        "tokenizers==0.22.1",
        "huggingface-hub==0.35.3",
    ),
    resources=flyte.Resources(
        cpu=110,
        memory="400Gi",
        # Trainium accelerator configuration
        gpu="Trn1:16",
    ),
    plugin_config=Elastic(
        nnodes=1,  # 1 Trainium instance (trn1.32xlarge)
        nproc_per_node=32,  # 32 NeuronCores per instance
    ),
    env_vars={
        "NEURON_RT_NUM_CORES": "32",  # Use all Neuron cores
        "NEURON_CC_FLAGS": "--model-type=transformer --distribution-strategy=llm-training --enable-mixed-precision-accumulation",
        "NEURON_COMPILE_CACHE_URL": "/var/tmp/neuron-compile-cache",  # Persistent cache
        "NEURON_FUSE_SOFTMAX": "1",  # Enable softmax fusion
        "NEURON_RT_STOCHASTIC_ROUNDING_EN": "1",  # Enable stochastic rounding for BF16
    },
    cache="auto",
)


pipeline_env = flyte.TaskEnvironment(
    name="bert-trainium-pipeline",
    image=flyte.Image.from_debian_base(name="bert-trainium-pipeline").with_pip_packages(
        "flyteplugins-pytorch==2.0.0b29"
    ),
    depends_on=[data_prep_env, trainium_env],
)


@dataclass
class DatasetConfig:
    """Configuration for FineWeb dataset loading and sampling"""

    shuffle: bool = True
    shuffle_seed: int = 42


@dataclass
class QuickTrainingConfig:
    """Training hyperparameters for BERT-Large Phase 1

    Default: QuickTest (for fast iteration)
    """

    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    warmup_steps: int = 10
    max_steps: int = 30
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    max_seq_length: int = 128

    # Logging and checkpointing
    lr_scheduler_type: str = "linear"
    logging_steps: int = 1
    save_steps: int = 10

    phase: int = 1


@dataclass
class MediumTrainingConfig(QuickTrainingConfig):
    """Medium training profile"""

    # Use larger batch size for real training
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8

    # Medium training schedule
    warmup_steps: int = 200
    max_steps: int = 3000
    gradient_accumulation_steps: int = 8

    # Logging
    logging_steps: int = 5
    save_steps: int = 50


@dataclass
class ProductionTrainingConfig(QuickTrainingConfig):
    """Production training profile"""

    # Use larger batch size for real training
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8

    # Production training schedule
    warmup_steps: int = 2000
    max_steps: int = 28125  # Phase 1: 90% of pretraining
    gradient_accumulation_steps: int = 64

    # Logging
    logging_steps: int = 10
    save_steps: int = 100


@dataclass
class QuickPhase2TrainingConfig(QuickTrainingConfig):
    """Phase 2: Fine-tuning with longer sequences"""

    # Phase 2 learning rate
    learning_rate: float = 7e-5

    # Batch size reduced due to longer sequences (memory constraints)
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 1

    max_seq_length: int = 512
    max_steps: int = 25
    warmup_steps: int = 5

    # Maximum predictions per sequence for MLM (80 = 15% of 512)
    max_pred_len: int = 80

    save_steps: int = 8
    phase: int = 2


@dataclass
class MediumPhase2Config(QuickPhase2TrainingConfig):
    """Medium Phase 2 profile"""

    gradient_accumulation_steps: int = 16
    max_steps: int = 500
    warmup_steps: int = 50  # 10% warmup


@dataclass
class ProductionPhase2Config(QuickPhase2TrainingConfig):
    """Production Phase 2 profile"""

    gradient_accumulation_steps: int = 512  # Global batch = 32,768
    max_steps: int = 1563  # Phase 2: remaining 10% of pretraining
    warmup_steps: int = 156  # 10% warmup


@data_prep_env.task
async def prepare_tokenized_dataset(
    fineweb_hf_path: str, num_samples: int, tokenizer_name: str, max_seq_length: int
) -> Dir:
    """
    Prepare FineWeb dataset by tokenizing and saving as PyTorch tensors.

    This task:
    1. Downloads FineWeb-Edu from HuggingFace
    2. Tokenizes the text data
    3. Saves as PyTorch tensors for efficient loading
    4. Uploads to blob storage
    """
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load FineWeb dataset
    print(f"Loading {num_samples:,} samples from FineWeb-Edu...")
    dataset = load_dataset(
        fineweb_hf_path,
        split="train",
        streaming=True,
    ).take(num_samples)

    output_path = f"fineweb-edu-seq{max_seq_length}"
    os.makedirs(output_path, exist_ok=True)

    print(f"Tokenizing with max_length={max_seq_length} and saving to {output_path}...")

    # Tokenize and save in batches
    batch_size = 10000
    current_batch = []
    batch_idx = 0
    batch_metadata = []  # Track batch sizes for efficient loading later

    for example in dataset:
        # Tokenize with specified max_seq_length
        tokens = tokenizer(
            example["text"],
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        current_batch.append(
            {
                "input_ids": tokens["input_ids"].squeeze(),
                "attention_mask": tokens["attention_mask"].squeeze(),
                "token_type_ids": tokens["token_type_ids"].squeeze(),
            }
        )

        # Save batch when it reaches batch_size
        if len(current_batch) >= batch_size:
            batch_data = {
                "input_ids": torch.stack([item["input_ids"] for item in current_batch]),
                "attention_mask": torch.stack(
                    [item["attention_mask"] for item in current_batch]
                ),
                "token_type_ids": torch.stack(
                    [item["token_type_ids"] for item in current_batch]
                ),
            }
            torch.save(batch_data, f"{output_path}/batch_{batch_idx}.pt")
            batch_metadata.append(
                {"batch_idx": batch_idx, "num_samples": len(current_batch)}
            )
            print(f"Saved batch {batch_idx} ({len(current_batch)} samples)")
            current_batch = []
            batch_idx += 1

    # Save remaining samples
    if current_batch:
        batch_data = {
            "input_ids": torch.stack([item["input_ids"] for item in current_batch]),
            "attention_mask": torch.stack(
                [item["attention_mask"] for item in current_batch]
            ),
            "token_type_ids": torch.stack(
                [item["token_type_ids"] for item in current_batch]
            ),
        }
        torch.save(batch_data, f"{output_path}/batch_{batch_idx}.pt")
        batch_metadata.append(
            {"batch_idx": batch_idx, "num_samples": len(current_batch)}
        )
        print(f"Saved final batch {batch_idx} ({len(current_batch)} samples)")

    # Save metadata file to avoid loading all batches during dataset initialization
    metadata_path = f"{output_path}/metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "batches": batch_metadata,
                "total_samples": sum(b["num_samples"] for b in batch_metadata),
            },
            f,
        )

    print(f"Dataset prepared at {output_path} ({batch_idx + 1} batches)")
    print(f"Metadata saved to {metadata_path}")

    remote_dir = await Dir.from_local(local_path=output_path)
    return remote_dir


class TokenizedDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for loading pre-tokenized data saved as .pt files with lazy loading."""

    def __init__(self, data_dir: str):
        """
        Initialize the dataset with lazy loading.
        """
        self.data_dir = data_dir
        self.batch_files = sorted(glob.glob(f"{data_dir}/batch_*.pt"))

        if not self.batch_files:
            raise ValueError(f"No batch files found in {data_dir}")

        print(f"Found {len(self.batch_files)} batch files")

        # Load metadata to avoid loading all batches
        metadata_path = os.path.join(data_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Build index from metadata without loading batch files
        self.sample_index = []
        for batch_info in metadata["batches"]:
            batch_idx = batch_info["batch_idx"]
            num_samples = batch_info["num_samples"]
            for sample_idx in range(num_samples):
                self.sample_index.append((batch_idx, sample_idx))

        print(f"Indexed {len(self.sample_index)} total samples from metadata.")

        # Cache for loaded batches (LRU-style, keep last N batches)
        # Cache size optimized for sequential access patterns
        self._batch_cache = {}
        self._cache_size = (
            10  # Keep 10 batches (100k samples) in memory for better throughput
        )

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        batch_idx, sample_idx = self.sample_index[idx]

        # Load batch if not in cache (with proper LRU eviction)
        if batch_idx not in self._batch_cache:
            # Evict oldest entry if cache is full
            if len(self._batch_cache) >= self._cache_size:
                # Remove least recently used (first item)
                oldest_key = next(iter(self._batch_cache))
                del self._batch_cache[oldest_key]

            # Load batch from disk
            batch_file = self.batch_files[batch_idx]
            self._batch_cache[batch_idx] = torch.load(batch_file, weights_only=True)
        else:
            # Move to end (mark as recently used) for LRU
            batch_data = self._batch_cache.pop(batch_idx)
            self._batch_cache[batch_idx] = batch_data

        batch_data = self._batch_cache[batch_idx]
        return {
            "input_ids": batch_data["input_ids"][sample_idx],
            "attention_mask": batch_data["attention_mask"][sample_idx],
            "token_type_ids": batch_data["token_type_ids"][sample_idx],
        }


@trainium_env.task(report=True)
def train_bert_on_trainium(
    dataset_path: Dir,
    training_config: QuickTrainingConfig,
    dataset_config: DatasetConfig,
    pretrained_model_name: str,
    compilation_cache: Optional[Dir] = None,
    resume_from_checkpoint: Optional[Dir] = None,
) -> tuple[Optional[Dir], Optional[Dir]]:
    """
    Distributed BERT pre-training on AWS Trainium instances.

    NOTE: Elastic plugin doesn't support async tasks, so this is a regular sync task.
    """
    import torch
    import torch.distributed as dist
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_backend
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from transformers import AutoTokenizer, BertForMaskedLM

    # Initialize distributed training
    dist.init_process_group("xla")  # Use XLA backend for Neuron
    rank = dist.get_rank()
    device = torch_xla.device()

    # Initialize metrics history (will be populated from checkpoint if resuming)
    resumed_metrics_history = []

    # Initialize Flyte report dashboard (only rank 0)
    if rank == 0:
        flyte.report.log(
            """
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
    .bert-dashboard { font-family: 'Segoe UI', sans-serif; }
    .bert-dashboard h1 { color: #333; }
    .bert-dashboard .metric-card {
        background: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .bert-dashboard .metric-value { font-size: 24px; font-weight: bold; color: #2196F3; }
    .bert-dashboard .metric-label { font-size: 14px; color: #666; margin-top: 5px; }
    .bert-dashboard .plot { width: 100%; height: 400px; }
</style>

<div class="bert-dashboard">
    <h1>🚀 BERT Trainium Training</h1>

    <!-- Current Metrics -->
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div class="metric-card">
            <div class="metric-value" id="current-step">0</div>
            <div class="metric-label">Current Step</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="current-loss">-</div>
            <div class="metric-label">Training Loss</div>
        </div>
    </div>

    <!-- Loss Plot -->
    <div class="metric-card">
        <h3>📈 Training Loss</h3>
        <div id="loss-plot" class="plot"></div>
    </div>
</div>

<script>
    // Initialize loss plot
    var lossTrace = {
        x: [], y: [], type: 'scatter', mode: 'lines+markers',
        name: 'Loss', line: {color: '#2196F3', width: 2},
        marker: {size: 6}
    };
    Plotly.newPlot('loss-plot', [lossTrace], {
        xaxis: {title: 'Step'},
        yaxis: {title: 'Loss'},
        showlegend: false
    });

    // Update function
    function updateMetrics(step, loss) {
        document.getElementById('current-step').textContent = step;
        document.getElementById('current-loss').textContent = loss.toFixed(4);

        Plotly.extendTraces('loss-plot', {
            x: [[step]],
            y: [[loss]]
        }, [0]);
    }
</script>
""",
            do_flush=True,
        )
        step_start_time = time.time()

    # Download compilation cache if provided (shared across all ranks)
    cache_dir = os.getenv("NEURON_COMPILE_CACHE_URL", "/var/tmp/neuron-compile-cache")
    os.makedirs(cache_dir, exist_ok=True)

    if rank == 0:
        if compilation_cache:
            print(f"[Rank 0] Downloading compilation cache to {cache_dir}...")
            compilation_cache.download_sync(local_path=cache_dir)
            print(f"[Rank 0] Compilation cache restored to {cache_dir}")
        else:
            print(f"[Rank 0] No compilation cache provided, will compile from scratch")

    # Wait for rank 0 to set up cache completely
    xm.rendezvous("cache_setup_complete")

    print(f"[Rank {rank}] Cache directory ready at {cache_dir}")

    # Download preprocessed dataset - ONLY rank 0 downloads to avoid redundant downloads
    # All 32 ranks share the same local NVMe disk, so only one needs to actually download
    if rank == 0:
        local_data_dir = dataset_path.download_sync(local_path="/tmp/flyte_dataset")
        print(f"[Rank 0] Dataset downloaded to {local_data_dir}")

    # Synchronize: wait for rank 0 to finish downloading before other ranks proceed
    xm.rendezvous("dataset_download_complete")

    if rank != 0:
        local_data_dir = "/tmp/flyte_dataset"
        print(f"[Rank {rank}] Using dataset at {local_data_dir}")

    dataset = TokenizedDataset(data_dir=local_data_dir)
    print(
        f"[Rank {rank}] Dataset created with {len(dataset)} samples (lazy-loaded with LRU cache)"
    )

    # Create DistributedSampler to ensure each rank sees different data.
    # The sampler handles:
    # 1. Shuffling the entire dataset with a shared seed
    # 2. Splitting shuffled indices across ranks (each rank gets different subset)
    world_size = dist.get_world_size()
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=dataset_config.shuffle,
        seed=dataset_config.shuffle_seed,
        drop_last=True,  # Ensure all ranks have same number of batches
    )

    if rank == 0:
        print(
            f"DistributedSampler: {len(dataset)} total samples, "
            f"{len(dataset) // world_size} samples per rank (world_size={world_size})"
        )

    # Use num_workers=0 to avoid worker process memory issues
    # With pre-tokenized data already saved as tensors, workers add overhead
    # without benefit.
    num_workers = 0
    dataloader = DataLoader(
        dataset,
        batch_size=training_config.per_device_train_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=False,  # No workers = no persistent workers
        prefetch_factor=None,  # Not used when num_workers=0
        pin_memory=False,  # Not needed for XLA device transfers
    )

    # Wrap with XLA parallel loader for Trainium distributed training
    # MpDeviceLoader handles:
    # - Transferring batches to XLA devices (NeuronCores)
    # - Buffered prefetching for overlapping compute and data transfer
    parallel_dataloader = pl.MpDeviceLoader(dataloader, device)
    print(
        f"[Rank {rank}] DataLoader ready with distributed sampling across {world_size} ranks"
    )

    # Load pretrained model or resume from checkpoint
    if resume_from_checkpoint:
        if rank == 0:
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")

        model = BertForMaskedLM.from_pretrained(pretrained_model_name)

        # Only rank 0 downloads checkpoint to avoid redundant downloads
        if rank == 0:
            local_checkpoint_path = resume_from_checkpoint.download_sync(
                local_path="/tmp/flyte_checkpoint"
            )
            print(f"[Rank 0] Checkpoint downloaded to {local_checkpoint_path}")

        # Wait for rank 0 to finish downloading
        xm.rendezvous("checkpoint_download_complete")

        if rank != 0:
            local_checkpoint_path = "/tmp/flyte_checkpoint"
            print(f"[Rank {rank}] Using checkpoint at {local_checkpoint_path}")

        checkpoint = torch.load(f"{local_checkpoint_path}/checkpoint.pt")

        model.load_state_dict(checkpoint["model_state_dict"])

        # Check if we're resuming the same phase or starting a new phase
        checkpoint_phase = checkpoint.get("phase", 1)
        if checkpoint_phase == training_config.phase:
            # Same phase: resume from checkpoint step
            start_step = checkpoint.get("step", 0)
            resumed_metrics_history = checkpoint.get("metrics_history", [])
            if rank == 0:
                print(f"Resuming Phase {training_config.phase} from step {start_step}")
        else:
            # Different phase: reset step counter, start fresh
            start_step = 0
            if rank == 0:
                print(
                    f"Starting Phase {training_config.phase} from Phase {checkpoint_phase} checkpoint (step counter reset to 0)"
                )
    else:
        if rank == 0:
            print(f"Loading pretrained model: {pretrained_model_name}")
        model = BertForMaskedLM.from_pretrained(pretrained_model_name)
        start_step = 0

    bert_config = model.config

    # Move model to BFloat16 for optimal Trainium performance
    # Loss will be kept in FP32 automatically by the model
    if rank == 0:
        print("Converting model to BFloat16 for Trainium optimization...")

    model = model.to(torch.bfloat16).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=(training_config.adam_beta1, training_config.adam_beta2),
        weight_decay=training_config.weight_decay,
        eps=training_config.adam_epsilon,
    )

    # Linear warmup scheduler
    def lr_lambda(current_step: int):
        if current_step < training_config.warmup_steps:
            return float(current_step) / float(max(1, training_config.warmup_steps))
        return max(
            0.0,
            float(training_config.max_steps - current_step)
            / float(max(1, training_config.max_steps - training_config.warmup_steps)),
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # If resuming from checkpoint, restore optimizer and scheduler state
    if resume_from_checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Fast-forward scheduler to correct step
        for _ in range(start_step):
            scheduler.step()
        if rank == 0:
            print(f"Restored optimizer and scheduler state to step {start_step}")

    model.train()
    global_step = start_step
    total_loss = 0.0
    output_dir = "bert_checkpoints"
    latest_remote_checkpoint = None
    latest_remote_cache_dir = None

    # Track metrics history for report resumption (restored from checkpoint if resuming)
    metrics_history = resumed_metrics_history.copy()

    # Restore historical metrics to the plot if resuming from checkpoint
    if rank == 0 and len(metrics_history) > 0:
        print(f"Restoring {len(metrics_history)} historical metrics to report...")
        for metric in metrics_history:
            flyte.report.log(
                f"""<script>updateMetrics({metric['step']}, {metric['loss']});</script>""",
                do_flush=False,
            )
        # Final flush after restoring all historical data
        flyte.report.log("", do_flush=True)
        print(f"Historical metrics restored to report")

    # Synchronize all ranks before training starts to ensure consistent model setup
    xm.rendezvous("model_setup_complete")
    if rank == 0:
        print("All ranks synchronized - starting training...")

    # Load tokenizer to get correct token IDs for MLM
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    mask_token_id = tokenizer.mask_token_id
    vocab_size = len(tokenizer)

    # Cache special token IDs for MLM masking
    special_token_ids = [
        tokenizer.pad_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
    ]
    special_token_ids = [tid for tid in special_token_ids if tid is not None]

    # Helper function for logging metrics to Flyte Report
    def log_training_metrics(
        step: int,
        loss_tensor,
        lr: float,
        step_time: float,
        logging_steps: int,
        batch_size: int,
        world_size: int,
    ):
        """Log training metrics to Flyte Report dashboard"""
        avg_loss = (loss_tensor / logging_steps).cpu().item()
        perplexity = math.exp(avg_loss)

        # Calculate throughput
        samples_per_sec = (logging_steps * batch_size * world_size) / step_time

        # Save metrics to history
        metrics_history.append(
            {
                "step": step,
                "loss": avg_loss,
                "perplexity": perplexity,
                "throughput": samples_per_sec,
            }
        )

        # Update Flyte Report (just send script to call updateMetrics)
        flyte.report.log(
            f"""<script>updateMetrics({step}, {avg_loss});</script>""",
            do_flush=True,
        )

        # Console logging
        print(
            f"Step {step}/{training_config.max_steps} | "
            f"Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | "
            f"LR: {lr:.2e} | {samples_per_sec:.1f} samples/s"
        )

        return time.time()  # Return new start time

    # Training loop
    for batch in parallel_dataloader:
        if global_step >= training_config.max_steps:
            # Mark step before breaking to ensure all pending operations complete
            xm.mark_step()
            break

        # Prepare masked language modeling inputs
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Create MLM labels following BERT paper strategy:
        # 1. Select 15% of tokens for masking
        # 2. Of those 15%:
        #    - 80% replace with [MASK]
        #    - 10% replace with random token
        #    - 10% keep unchanged
        labels = input_ids.clone()

        # Create random array for selecting tokens (15% of all tokens)
        probability_matrix = torch.full(input_ids.shape, 0.15, device=input_ids.device)

        # Don't mask special tokens (PAD, CLS, SEP)
        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for special_token_id in special_token_ids:
            special_tokens_mask |= input_ids == special_token_id
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # 80% of the time: replace with [MASK]
        indices_replaced = (
            torch.bernoulli(
                torch.full(input_ids.shape, 0.8, device=input_ids.device)
            ).bool()
            & masked_indices
        )
        input_ids[indices_replaced] = mask_token_id

        # 10% of the time: replace with random token
        indices_random = (
            torch.bernoulli(
                torch.full(input_ids.shape, 0.5, device=input_ids.device)
            ).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            vocab_size, input_ids.shape, dtype=torch.long, device=input_ids.device
        )
        input_ids[indices_random] = random_words[indices_random]

        # Remaining 10%: keep original token (no change needed)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss / training_config.gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Accumulate loss (detach to avoid growing computation graph)
        total_loss += loss.detach()

        # Gradient accumulation
        if (global_step + 1) % training_config.gradient_accumulation_steps == 0:
            # Mark step to execute backward pass on XLA device
            xm.mark_step()

            # Reduce gradients across all ranks
            xm.reduce_gradients(optimizer)

            # Gradient clipping (after reduction)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), training_config.max_grad_norm
            )

            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1

        # Logging
        if global_step % training_config.logging_steps == 0:
            # Reduce loss across all ranks for accurate logging
            total_loss_reduced = xm.all_reduce(
                xm.REDUCE_SUM,
                total_loss,
                scale=1.0 / world_size,  # Average across ranks
            )

            if rank == 0:

                def log_fn():
                    nonlocal step_start_time

                    step_time = time.time() - step_start_time
                    lr = scheduler.get_last_lr()[0]

                    # Log all metrics and get new start time
                    step_start_time = log_training_metrics(
                        step=global_step,
                        loss_tensor=total_loss_reduced,
                        lr=lr,
                        step_time=step_time,
                        logging_steps=training_config.logging_steps,
                        batch_size=training_config.per_device_train_batch_size,
                        world_size=world_size,
                    )

                if global_step % (training_config.logging_steps * 10) == 0:
                    try:
                        result = subprocess.run(
                            ["neuron-ls"],
                            capture_output=True,
                            text=True,
                            timeout=3,
                        )
                        if result.returncode == 0:
                            print(f"\n{'='*60}")
                            print(f"NeuronCore Status at Step {global_step}")
                            print("=" * 60)
                            print(result.stdout)
                            print("=" * 60)
                    except subprocess.TimeoutExpired:
                        print(f"neuron-ls timed out at step {global_step}")
                    except FileNotFoundError:
                        print("neuron-ls not found (skipping neuron monitoring)")
                    except Exception as e:
                        print(f"Failed to run neuron-ls: {e}")

                # Use step closure to defer CPU operations and avoid blocking
                xm.add_step_closure(log_fn)

            total_loss = 0.0

        # Checkpoint saving
        if global_step % training_config.save_steps == 0:
            # Execute all pending operations before checkpointing to ensure graph consistency
            xm.mark_step()

            if rank == 0:
                # Create checkpoint directory
                checkpoint_path = f"{output_dir}/checkpoint-{global_step}"
                os.makedirs(checkpoint_path, exist_ok=True)

                # Convert XLA tensors to CPU (after mark_step, so no pending ops)
                model_state = xm._maybe_convert_to_cpu(model.state_dict())
                optimizer_state = xm._maybe_convert_to_cpu(optimizer.state_dict())

                # Save checkpoint
                torch.save(
                    {
                        "model_state_dict": model_state,
                        "optimizer_state_dict": optimizer_state,
                        "step": global_step,
                        "phase": training_config.phase,
                        "metrics_history": metrics_history.copy(),
                    },
                    f"{checkpoint_path}/checkpoint.pt",
                )

                # Save model config
                bert_config.save_pretrained(checkpoint_path)

                # Upload checkpoint
                print(f"Uploading checkpoint {checkpoint_path}...")
                latest_remote_checkpoint = Dir.from_local_sync(
                    local_path=checkpoint_path,
                    remote_destination=f"s3://bert-trainium-aws/{flyte.ctx().action.run_name}/checkpoints/phase-{training_config.phase}/step-{global_step}",
                )
                latest_remote_cache_dir = Dir.from_local_sync(
                    local_path=cache_dir,
                    remote_destination=f"s3://bert-trainium-aws/{flyte.ctx().action.run_name}/neuron-compile-cache/phase-{training_config.phase}/step-{global_step}",
                )

                print(f"✓ Checkpoint uploaded for step {global_step}")

    # Execute any pending XLA operations
    xm.mark_step()

    # Synchronize all ranks before cleanup
    xm.rendezvous("training_complete")

    if rank == 0:
        print(f"\n✓ Training loop completed at step {global_step}")
        print(
            f"Latest checkpoint: {latest_remote_checkpoint.path if latest_remote_checkpoint else 'N/A'}"
        )

    # Cleanup resources
    print(f"[Rank {rank}] Cleaning up resources...")

    del parallel_dataloader
    del dataloader
    model = model.cpu()
    del model
    del optimizer

    xm.mark_step()
    xm.wait_device_ops()
    dist.destroy_process_group()

    print(f"[Rank {rank}] Cleanup complete")

    return latest_remote_checkpoint, latest_remote_cache_dir


@pipeline_env.task
async def bert_trainium_pipeline(
    fineweb_path: str,
    pretrained_model_name: str,
    training_config: QuickTrainingConfig,
    dataset_config: DatasetConfig,
    num_samples: Optional[int] = None,
    compilation_cache: Optional[Dir] = None,
    resume_from_checkpoint: Optional[Dir] = None,
) -> tuple[Optional[Dir], Optional[Dir]]:
    """
    End-to-end BERT pre-training pipeline on AWS Trainium.
    """
    # Determine phase and sequence length
    phase = training_config.phase
    max_seq_length = training_config.max_seq_length

    print(f"Running Phase {phase} training (sequence length: {max_seq_length})")

    # Calculate required samples if not provided (just a simple heuristic)
    # Samples needed = max_steps × gradient_accumulation_steps × per_device_batch_size × world_size
    if num_samples is None:
        world_size = int(os.getenv("NEURON_RT_NUM_CORES", "32"))
        num_samples = (
            training_config.max_steps
            * training_config.gradient_accumulation_steps
            * training_config.per_device_train_batch_size
            * world_size
        )
        print(
            f"Auto-calculated num_samples: {num_samples:,} (based on training config)"
        )

    # Step 1: Prepare tokenized dataset
    print(
        f"Preparing FineWeb dataset with tokenization (max_seq_length={max_seq_length})..."
    )
    dataset_path = await prepare_tokenized_dataset(
        fineweb_hf_path=fineweb_path,
        num_samples=num_samples,
        tokenizer_name="bert-large-uncased",
        max_seq_length=max_seq_length,
    )

    # Step 2: Distributed training on Trainium
    print(
        f"Starting BERT Phase {phase} training from {pretrained_model_name} on 1 Trainium instance (32 NeuronCores)..."
    )

    try:
        model_dir, cache_dir = train_bert_on_trainium(
            dataset_path=dataset_path,
            training_config=training_config,
            dataset_config=dataset_config,
            pretrained_model_name=pretrained_model_name,
            compilation_cache=compilation_cache,
            resume_from_checkpoint=resume_from_checkpoint,
        )

        print(f"\n✓ Training completed successfully!")
        print(f"Model checkpoint: {model_dir.path if model_dir else 'N/A'}")
        print(f"Compilation cache: {cache_dir.path if cache_dir else 'N/A'}")

        return model_dir, cache_dir
    except flyte.errors.RuntimeUserError as e:
        # Training failed
        print(f"\n{'='*80}")
        print(f"⚠️  TRAINING FAILED")
        print(f"{'='*80}")
        print(
            "Latest checkpoint: "
            + f"s3://bert-trainium-aws/{flyte.ctx().action.run_name}/checkpoints/phase-{training_config.phase}/"
        )
        print(
            "Latest compilation cache: "
            + f"s3://bert-trainium-aws/{flyte.ctx().action.run_name}/neuron-compile-cache/phase-{training_config.phase}/"
        )

        raise e


@pipeline_env.task
async def bert_two_phase_pipeline(
    fineweb_path: str = "HuggingFaceFW/fineweb-edu",
    pretrained_model_name: str = "bert-large-uncased",
    dataset_config: DatasetConfig = DatasetConfig(),
    phase1_config: QuickTrainingConfig = QuickTrainingConfig(),
    phase2_config: QuickPhase2TrainingConfig = QuickPhase2TrainingConfig(),
    phase1_compilation_cache: Optional[Dir] = None,
    phase2_compilation_cache: Optional[Dir] = None,
    phase1_checkpoint: Optional[Dir] = None,
    phase2_checkpoint: Optional[Dir] = None,
) -> tuple[Optional[Dir], Optional[Dir], Optional[Dir]]:
    """
    Complete two-phase BERT pre-training pipeline.
    """
    # ========================================================================
    # Phase 1: 128 token sequences
    # ========================================================================
    print("=" * 80)
    print("Starting Phase 1: 128 token sequences")
    print("=" * 80)

    phase1_final_checkpoint, phase1_cache = await bert_trainium_pipeline(
        fineweb_path=fineweb_path,
        pretrained_model_name=pretrained_model_name,
        training_config=phase1_config,
        dataset_config=dataset_config,
        compilation_cache=phase1_compilation_cache,
        resume_from_checkpoint=phase1_checkpoint,
    )

    print("Phase 1 complete!")
    print(
        f"Phase 1 checkpoints saved to: {phase1_final_checkpoint.path if phase1_final_checkpoint else 'N/A'}"
    )
    print(
        f"Phase 1 compilation cache saved to: {phase1_cache.path if phase1_cache else 'N/A'}"
    )

    # ========================================================================
    # Phase 2: 512 token sequences (resume from Phase 1)
    # ========================================================================
    print("=" * 80)
    print("Starting Phase 2: 512 token sequences")
    print("=" * 80)

    phase2_checkpoint, phase2_cache = await bert_trainium_pipeline(
        fineweb_path=fineweb_path,
        pretrained_model_name=pretrained_model_name,
        training_config=phase2_config,
        dataset_config=dataset_config,
        compilation_cache=phase2_compilation_cache,
        resume_from_checkpoint=phase2_checkpoint or phase1_final_checkpoint,
    )

    print("Phase 2 complete!")
    print(
        f"Final model saved to: {phase2_checkpoint.path if phase2_checkpoint else 'N/A'}"
    )
    print(
        f"Phase 2 compilation cache (seq_len=512): {phase2_cache.path if phase2_cache else 'N/A'}"
    )

    return phase2_checkpoint, phase1_cache, phase2_cache


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(bert_two_phase_pipeline)
    print(f"Run URL: {run.url}")
