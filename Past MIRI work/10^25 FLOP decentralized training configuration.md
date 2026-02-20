# Distributed training configuration: 10^25 FLOP

There are a few configurations of hardware that would achieve a training run of this size without exotic or speculative engineering or algorithmic innovations, generally sticking to techniques that are already demonstrated to work at this scale, and without using hardware that exceeds the covered compute cluster threshold in any one node.

Either of the following node configurations (as well as many others) would suffice:

- 48x A100  
- 16x GH200

With each node sized at slightly under the CCC threshold, you’d need around 70 non-covered clusters to accomplish 10^25 FLOP of training (after accounting for reduced MFU) within 1 year. Having more nodes would complete this amount of training faster, but increase the risk of detection and reduce the efficiency of training.[^1]

For simplicity I’ll present the configuration using 72 nodes of 16x GH200 NVL2 with 100 Mbps WAN bandwidth. This should still work with other types of GPUs and/or moderately slower connections with only minor adjustments.

## Model and dataset size

Two considerations here:

1. How large of a model can fit into the node’s memory?  
   * We should try to keep the model small enough that it fits into the memory of a single node. Otherwise, we’ll have to shard the model across multiple nodes, which requires more communication and more engineering to implement other distributed training techniques.  
2. How large of a model is optimal to train with this much compute?  
   * The Chinchilla study established that there is an optimal ratio of parameter count to number of datapoints in the pretraining dataset, for minimizing language modeling loss given a fixed compute budget. If you use too many parameters, you could get better performance by training a smaller model on more data, and if you use too much training data, you could improve performance by training a bigger model on less data.  
     * The constant of proportionality varies with model architecture and details of the training data mix, but the important takeaway is that dataset size must be scaled proportional to parameter count if you want to optimize performance relative to compute expenditure.

With those considerations in mind, how large of a model should we train to optimally use 10^25 FLOP of compute, and with the constraint of fitting into one node?

Consider our hardware, the GB200 NVL2. We have nodes of 16 cards, which is at the limit of performance for non-covered compute clusters. It has 144 GB of HBM3e memory per GPU, for a total of 2304 GB of memory per node.

Let’s conservatively assume that the model is in FP16 numeric format[^2], so each parameter takes two bytes in memory. Then the node can run inference on dense models of up to 1152B parameters, and FP16 training of dense models up to **144B parameters**.[^3]

The Chinchilla-optimal dataset size for a model of that size is:  
25 tokens/parameter \* 144B parameters \= 3.6T tokens

The Chinchilla-optimal compute for a model of that size is:  
6 FLOP/token/parameter \* (3.6T tokens) \* (144B parameters) \= 3e24 FLOP

Chinchilla-optimality is the correct target to aim for if you don’t consider inference costs, and you only care about having the lowest possible loss at the end of training. In practice, you will use the model at least once, or most likely, many times until its lifetime inference cost is on the order of your upfront training cost. So in practice it’s best to “overtrain” models relative to the Chinchilla scaling law, i.e. use more datapoints than the scaling law prescribes for optimizing only for minimum loss. I won’t get into this here, but let’s \~3.3x our dataset size, bringing the **compute** to **1.02e25 FLOP** and the **dataset** to **12T tokens**.

How can we complete that training run in the shortest possible time with our given hardware and network?

## Training configuration

The simplest and fastest way to achieve this is to **avoid multi-node model parallelism**, since we are limited to 100 Mbps of inter-node bandwidth.

Pipeline parallelism across WAN is possible (e.g. using techniques from the SWARM paper and compression of activations between layers), but this requires more engineering, and AFAIK it hasn’t been demonstrated at over 1% of this scale. We don’t need it for this case, although it might be necessary for distributed training runs on the order of 1e26 FLOP or larger.

### Recommended topology

**Inside each node**, use a standard ML stack: tensor and pipeline parallelism, ZeRO, and 16-bit weights.  
**Across the 72 nodes**, treat each node as a replica and use the DiLoCo technique of decentralized data parallelism, where each node locally does multiple inner steps, and then occasional synchronization on outer steps.

### Intra-node configuration

Within nodes, we have fast interconnect and large memory capacity, similar to what is available in a single-datacenter cluster. We’ll use tensor and pipeline parallelism across GPUs within the nodes, which has already been optimized and demonstrated in many papers.

For a 144B model across 16 GPUs, a good configuration is:

* Tensor Parallel (TP): 8×  
  * keeps matmul shards small and fast; easily accommodated by NVLink  
* Pipeline Parallel (PP): 2×  
  * helps the model fit into local memory and keeps pipeline bubbles small, for high HFU  
* Data Parallel (DP): 1× (within node)  
  * We’ll use data parallelism, but *across* nodes instead of *within* nodes.

#### Memory/optimizer choices

* Use **FP8 or BF16 mixed precision** (GH200 supports very high Tensor Core throughput for BF16/FP16/FP8).  
* Use **activation checkpointing** (recompute) and FlashAttention-class kernels.  
* Use **sharded optimizer states** (ZeRO-2/ZeRO-3 or FSDP) *within node* to keep per-GPU memory comfortable at 140–160B.

### Inter-node configuration

We want to avoid synchronizing every node on every step.  
If each step takes 25 seconds, 100 Mbps of bandwidth is only enough to synchronize models up to 39M parameters in FP32 or 78M params in FP16.[^4] So vanilla data parallelism, like we would do if all the GPUs were in a single cluster, will not work over WAN.

Instead, we should use streaming DiLoCo and compression.

DiLoCo enables up to \~500 inner steps with only a small performance penalty, giving up to \~500x bandwidth reduction. Let’s use 128 inner steps per outer step, for an outer step time of around 50 minutes.

Gradient communication can be reduced further with quantization and sparsification, while error-feedback accumulation minimizes the accuracy losses from these approximations.  
Streaming DiLoCo sequentially synchronizes different subsets of parameters and overlaps compute with communication. In their paper they combined with 4-bit quantization and achieved a 100x reduction in bandwidth.

Treating each of 72 nodes as a diloco worker, here are the settings:  
outer interval: \~53 minutes  
inner steps: 25 seconds; 128 inner steps per outer step  
ChatGPT recommendation: 4096 sequence length, 32 sequences \=\> 131k tokens per local batch \=\> 23 sec per step  
Robi still verifying this math. (Why is this batch size optimal?)

Compression settings:

- 4-bit quantization  
- 25% sparsification  
- error-feedback accumulation

Streaming mechanics:

- split parameters into buckets, e.g. 512 MB  
- maintain a communication thread that continuously:  
  - gets the latest bucket deltas / accumulated gradients  
  - quantizes and sparsifies,  
  - sends to aggregator  
  - receives the averaged bucket update  
  - applies it to the local model

Aggregation topology:  
In a  cloud server, collect the gradients from each node, average them, and send the result back to each node.

- You can use 2-3 of these for redundancy.  
- You need at least 100 Mbps \* 72 \= 7.2 Gbps of bandwidth here, which is a bit more than the server for a medium-traffic website but tiny compared to datacenters.  
- You can apply asynchronous collection with a cutoff. For example, apply an outer update when ≥64/72 nodes have contributed for that “round,” and carry late arrivals into the next round with appropriate weighting. This way, stragglers or dead nodes won’t slow down the rest of the job.

### Post-training

Won’t elaborate on this too much for now. Post-training is the easy part.

They’ll probably want to do instruction tuning, and RL for reasoning training. Asynchronous RL is easy because you can overlap rollout generation with training.

[^1]:  “This process was demonstrated in the DiLoCo paper by Douillard et al. (2023), building on previous work, such as that by Sticht et al. (2018). This process is not equivalent to the usual training process, and as such it can harm performance. However, the DiLoCo authors find that they can conduct training with up to 500 inner steps with only a

[^2]:  In practice, training has already started moving to smaller numeric formats, which will allow even larger models to be trained on each node.

[^3]:  RR to-do: explain 16 bytes per parameter assumption for FP16 model training. For models in smaller formats, the memory requirement is less, and you can train a bigger model.

[^4]:  2x send+receive \* model size \= time x bandwidth \=\> 10 minutes \* 100 Mbps / (2 \* 4 bytes/param) \= 938M params