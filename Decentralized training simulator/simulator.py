import json
import argparse
import math
import os

class Model:
    def __init__(self, parameters, hidden_size=4096, precision_bytes=2):
        self.parameters = parameters
        self.hidden_size = hidden_size
        self.precision_bytes = precision_bytes

    @property
    def model_size_bytes(self):
        return self.parameters * self.precision_bytes

class Cluster:
    def __init__(self, num_nodes, flop_per_node, memory_per_node, bandwidth_bps, mfu=0.4):
        self.num_nodes = num_nodes
        self.flop_per_node = flop_per_node
        self.memory_per_node = memory_per_node # in Bytes
        self.bandwidth_bps = bandwidth_bps # bits per second
        self.mfu = mfu # Model Flop Utilization

    @property
    def total_compute_flop_s(self):
        return self.num_nodes * self.flop_per_node * self.mfu

class Algorithm:
    def __init__(self, name, inner_steps=1, compression_ratio=1.0, gradient_accumulation_steps=1):
        self.name = name
        self.inner_steps = inner_steps
        self.compression_ratio = compression_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def communication_volume_per_step(self, model):
        """
        Returns the communication volume in bits for one global step.
        """
        if self.name == "DiLoCo":
            # 2 * model_size (send + receive) * 8 bits/byte
            # Divided by inner_steps because comms happen once every inner_steps
            # Adjusted by compression
            base_comm = 2 * model.model_size_bytes * 8
            return (base_comm / self.inner_steps) / self.compression_ratio
        elif self.name == "DDP":
            # Standard DDP: 2 * model_size per step (Ring AllReduce)
            return 2 * model.model_size_bytes * 8
        else:
            raise ValueError(f"Unknown algorithm: {self.name}")

def calculate_training_time(model, dataset_tokens, cluster, algorithm):
    # Total FLOPs required
    # Approximation: 6 * parameters * tokens
    total_flops = 6 * model.parameters * dataset_tokens
    
    # Compute Time
    compute_time_seconds = total_flops / cluster.total_compute_flop_s
    
    # Communication Time
    # Total steps = Total tokens / (Batch Size * Sequence Length) 
    # But for simplicity, we can calculate throughput.
    
    # Let's look at it per-step.
    # We need to assume a batch size or tokens per step.
    # From the 10^25 config: "131k tokens per local batch => 23 sec per step"
    # This implies a specific tokens/sec per node.
    
    # Let's derive steps from compute.
    # FLOPs per step (per node) approx = 6 * parameters * tokens_per_step_per_node
    # We don't have tokens_per_step in the generic call, so let's model bandwidth vs compute rate.
    
    # Bandwidth intensity (arithmetic intensity inverted): Bits communicated per FLOP.
    # Comm / Comp = (Comm Volume / Step) / (FLOPs / Step)
    
    # Let's stick to the high-level summation:
    # Total Compute Time = Total FLOPs / (Nodes * Node_FLOPs * MFU)
    
    # Total Comm Time:
    # Total Global Steps = Total Tokens / Global Batch Size
    # We need Global Batch Size.
    
    # Let's infer Global Batch Size from the "131k tokens per local batch" in the config if provided,
    # or allow it to be passed.
    
    # For this version, let's assume perfect overlap is NOT possible (pessimistic)
    # or perfectly masked (optimistic). The report says "overlap compute with communication".
    
    # Simplified DiLoCo Model:
    # Time = max(Compute, Comm) if perfectly overlapped
    # Time = Compute + Comm if no overlap
    
    # Let's calculate rates:
    # Compute Rate (Tokens/sec) = Cluster_FLOPs / (6 * Params)
    # Comm Rate limit (Tokens/sec) = (Bandwidth / Bits_per_token_comm) * Nodes ? 
    # Careful with topology. 
    # In DiLoCo with a central reducer (as per config):
    # Each node sends gradients to center and receives updates. 
    # Bottleneck is usually the node's WAN link (100 Mbps).
    # So each node must transmit (Comm_Volume_Per_Step / Num_Nodes?) -> No, DiLoCo sends full model usually?
    # Wait, DiLoCo sends the *pseudo-gradient* which is the size of the model (or params).
    # Each node sends its accumulator.
    
    # Comm Volume per Node per Outer Step = 2 * Model_Size (Send + Recv) * Compression
    # Time for Comm per Outer Step = Volume / Node_Bandwidth
    
    # Compute Time per Outer Step = Inner_Steps * Time_Per_Inner_Step
    
    # We need 'Time_Per_Inner_Step'.
    # Time_Per_Inner_Step = (6 * Params * Local_Batch_Tokens) / (Node_FLOPs * MFU)
    
    return compute_time_seconds, 0 # Placeholder for now

def simulate_diloco(config):
    # Extract config
    model_conf = config['model']
    hardware_conf = config['hardware']
    algo_conf = config['algorithm']
    training_conf = config['training']
    
    # Objects
    model = Model(
        parameters=model_conf['parameters'],
        precision_bytes=model_conf.get('precision_bytes', 2)
    )
    
    cluster = Cluster(
        num_nodes=hardware_conf['num_nodes'],
        flop_per_node=hardware_conf['flop_per_node'],
        memory_per_node=hardware_conf['memory_per_node'],
        bandwidth_bps=hardware_conf['bandwidth_bps'],
        mfu=hardware_conf.get('mfu', 0.4)
    )
    
    # DiLoCo specifics
    inner_steps = algo_conf['inner_steps']
    local_batch_tokens = training_conf['local_batch_tokens']
    total_tokens = training_conf['total_tokens']
    compression_ratio = algo_conf.get('compression_ratio', 1.0)
    
    # Calculations
    
    # 1. Compute Time per Inner Step (per node)
    # FLOPs per inner step = 6 * P * Local_Tokens
    flops_per_step = 6 * model.parameters * local_batch_tokens
    node_compute_power = cluster.flop_per_node * cluster.mfu
    time_per_inner_step = flops_per_step / node_compute_power
    
    # 2. Communication Time per Outer Step (per node)
    # Transmit Model Delta + Receive Averaged Model
    # Size = Parameters * Precision * Compression
    # In DiLoCo, you exchange the full model delta.
    payload_bits = (model.model_size_bytes * 8) / compression_ratio
    # Round trip (Up + Down)
    total_comm_bits = 2 * payload_bits
    
    comm_time_per_outer_step = total_comm_bits / cluster.bandwidth_bps
    
    # 3. Total Time
    # One Outer Step = 'inner_steps' of Compute + 1 Comm phase
    # Duration Outer Step = (inner_steps * time_per_inner_step) + comm_time_per_outer_step
    # (Assuming NO overlap for conservative estimate, or subtract overlap if 'streaming' is on)
    
    if algo_conf.get('streaming', False):
        # If streaming, comms happens in background of compute.
        # Effective time is max(Compute_Block, Comm_Block)
        # But usually you need the weight update before the next outer step? 
        # Standard DiLoCo: Sync is blocking.
        # Streaming DiLoCo: Overlaps.
        # Let's stick to Blocking DiLoCo for the "10^25" config as baseline, 
        # but the doc mentions "Streaming DiLoCo".
        # If streaming, we effectively hide the comms if Compute > Comm.
        compute_block_time = inner_steps * time_per_inner_step
        effective_outer_time = max(compute_block_time, comm_time_per_outer_step)
    else:
        effective_outer_time = (inner_steps * time_per_inner_step) + comm_time_per_outer_step

    # Total Steps
    global_batch_tokens = local_batch_tokens * cluster.num_nodes
    total_global_steps = total_tokens / global_batch_tokens
    total_outer_steps = total_global_steps / inner_steps
    
    total_time_seconds = total_outer_steps * effective_outer_time
    
    # Formatting
    days = total_time_seconds / (24 * 3600)
    
    results = {
        "compute_time_per_inner_step": time_per_inner_step,
        "comm_time_per_outer_step": comm_time_per_outer_step,
        "total_time_days": days,
        "feasibility": "Feasible" if days < 365 else "Impractical (>1 year)",
        "bottleneck": "Communication" if comm_time_per_outer_step > (inner_steps * time_per_inner_step) else "Compute"
    }
    return results

def main():
    parser = argparse.ArgumentParser(description="Decentralized Training Simulator")
    parser.add_argument("--config", type=str, required=True, help="Path to config file (JSON)")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    print(f"--- Simulation Configuration: {args.config} ---")
    print(json.dumps(config, indent=2))
    print("\n--- Results ---")
    
    if config['algorithm']['name'] == 'DiLoCo':
        results = simulate_diloco(config)
        print(json.dumps(results, indent=2))
    else:
        print("Algorithm not yet implemented.")

if __name__ == "__main__":
    main()
