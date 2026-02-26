import { useState, useEffect } from 'react'

const Tooltip = ({ text }: { text: string }) => (
  <div className="tooltip-container">
    ⓘ
    <span className="tooltip-text">{text}</span>
  </div>
);

const GrowthInput = ({ label, value, onChange, tooltip, unit }: { label: string, value: number, onChange: (val: number) => void, tooltip: string, unit: string }) => {
  const inputStyle = {
    background: '#2a2a2a',
    color: '#e2e8f0',
    border: '1px solid #475569',
    padding: '6px 10px',
    borderRadius: '6px',
    width: '100px',
    fontSize: '0.9em',
    outline: 'none',
    boxSizing: 'border-box' as const
  };

  // Convert internal OOM/yr to display value
  const displayValue = unit === 'oom'
    ? parseFloat(value.toFixed(3))
    : unit === 'percent'
    ? Math.round((Math.pow(10, value) - 1) * 100)
    : parseFloat(Math.pow(10, value).toFixed(2));

  const step = unit === 'oom' ? 0.01 : unit === 'percent' ? 1 : 0.1;

  const handleChange = (raw: number) => {
    if (unit === 'oom') onChange(raw);
    else if (unit === 'percent') onChange(Math.log10(raw / 100 + 1));
    else onChange(Math.log10(raw)); // multiple
  };

  return (
    <div style={{ display: 'flex', gap: '12px', alignItems: 'center', marginBottom: '10px' }}>
      <label style={{ fontSize: '0.9em', fontWeight: 500, color: '#94a3b8', width: '140px', flexShrink: 0 }}>
        {label}
        <Tooltip text={tooltip} />
      </label>
      <input
        type="number"
        step={step}
        value={displayValue}
        onChange={(e) => handleChange(Number(e.target.value))}
        style={inputStyle}
      />
    </div>
  );
};

function App() {
  // Model Parameters
  const [parameters, setParameters] = useState(144e9) // 144B
  const [tokens, setTokens] = useState(12e12) // 12T
  const [isMoE, setIsMoE] = useState(false)
  const [activeParams, setActiveParams] = useState(24e9) // 24B
  const [moeLayers, setMoeLayers] = useState(32)
  const [expertParallelism, setExpertParallelism] = useState('none') // none, regional, global
  
  // Hardware Parameters
  const [numNodes, setNumNodes] = useState(72)
  const [pflopsPerNode, setPflopsPerNode] = useState(16) // Sub-CCC node (≤16 H100-equiv)
  const [vramPerNode, setVramPerNode] = useState(2304) // 16x 144GB
  const [bandwidthMbps, setBandwidthMbps] = useState(100)
  const [latencyMs, setLatencyMs] = useState(100) // Inter-node ping
  const [mfu, setMfu] = useState(0.4)

  // Hierarchical Parameters
  const [useHierarchy, setUseHierarchy] = useState(false)
  const [nodesPerGroup, setNodesPerGroup] = useState(8)
  const [regionalBandwidth, setRegionalBandwidth] = useState(1000) // 1 Gbps
  const [regionalLatency, setRegionalLatency] = useState(20) // 20ms
  const [regionalSteps, setRegionalSteps] = useState(16)

  // Algorithm Parameters
  const [innerSteps, setInnerSteps] = useState(128)
  const [compression, setCompression] = useState(16)
  const [localBatch, setLocalBatch] = useState(131072)
  const [ppCompression, setPpCompression] = useState(10)
  const [microBatches, setMicroBatches] = useState(8)
  const [precision, setPrecision] = useState('FP16')
  const [streamingEnabled, setStreamingEnabled] = useState(true)

  // Straggler Mitigation
  const [stragglerStrategy, setStragglerStrategy] = useState('none') // none, threshold, redundancy

  // Maximum Training Duration
  const [showMaxDuration, setShowMaxDuration] = useState(false)
  const [growthUnit, setGrowthUnit] = useState('percent') // 'oom', 'percent', 'multiple'
  const [hwGrowth, setHwGrowth] = useState(Math.log10(1.37))
  const [swGrowth, setSwGrowth] = useState(Math.log10(3))
  const [investGrowth, setInvestGrowth] = useState(Math.log10(3.5))
  const [manualMaxDays, setManualMaxDays] = useState(false)
  const [manualMaxDaysValue, setManualMaxDaysValue] = useState(180)

  const [results, setResults] = useState<any>(null)

  useEffect(() => {
    calculate()
  }, [parameters, tokens, numNodes, pflopsPerNode, vramPerNode, bandwidthMbps, latencyMs, mfu, innerSteps, compression, localBatch, ppCompression, microBatches, useHierarchy, nodesPerGroup, regionalBandwidth, regionalLatency, regionalSteps, hwGrowth, swGrowth, investGrowth, stragglerStrategy, streamingEnabled, isMoE, activeParams, moeLayers, expertParallelism, manualMaxDays, manualMaxDaysValue, precision])

  const calculate = () => {
    // 1. Memory Analysis
    const bytesPerParam = precision === 'FP16' ? 16 : precision === 'FP8' ? 14 : 13
    const bytesPerValue = precision === 'FP16' ? 2 : precision === 'FP8' ? 1 : 0.5

    // For MoE with EP: each node stores shared params + (expert params / EP degree)
    const expertParams = isMoE ? (parameters - activeParams) : 0
    const sharedParams = parameters - expertParams
    const epDegree = (isMoE && expertParallelism !== 'none')
      ? (expertParallelism === 'global' ? numNodes : nodesPerGroup)
      : 1
    const perNodeParams = sharedParams + expertParams / epDegree
    const memoryBytes = perNodeParams * bytesPerParam
    const isSharded = memoryBytes > (vramPerNode * 1e9)
    const ppStages = Math.ceil(memoryBytes / (vramPerNode * 1e9))
    
    // 2. Resource Adjustments (e.g. Redundancy / Backup Workers)
    const effectiveNodes = stragglerStrategy === 'redundancy' ? numNodes / 1.1 : numNodes

    // 3. Compute Time
    // Compute depends on ACTIVE parameters for MoE
    const computeParams = isMoE ? activeParams : parameters
    const flopsPerStep = 6 * computeParams * localBatch
    const nodeComputePower = pflopsPerNode * 1e15 * mfu
    
    // Expert Parallelism (All-to-All) Latency Penalty
    let epLatencySec = 0
    if (isMoE) {
      if (expertParallelism === 'global') {
        epLatencySec = (latencyMs / 1000) * 2 * moeLayers
      } else if (expertParallelism === 'regional' && useHierarchy) {
        epLatencySec = (regionalLatency / 1000) * 2 * moeLayers
      }
    }

    const computeTimePerStep = (flopsPerStep / nodeComputePower) + epLatencySec
    
    // 4. Algorithmic Efficiency Penalty
    // Research (Wang et al. 2018) suggests hierarchy "anchors" drift.
    const effectiveH = useHierarchy 
      ? innerSteps * Math.pow(regionalSteps, 0.5) // Hierarchical benefit (drift anchoring)
      : innerSteps
    
    // Alpha reduces slightly for larger models (more robust)
    const baseAlpha = 0.08 * (1 / (1 + Math.log10(parameters / 1e9) / 5))
    // Strategy: Threshold aggregation is faster but less efficient per token (staleness)
    const strategyPenalty = stragglerStrategy === 'threshold' ? 1.15 : 1.0
    const algorithmicEfficiency = Math.max(0.4, (1 - baseAlpha * Math.log10(effectiveH)) / strategyPenalty)
    
    // 5. Straggler & Congestion Penalty
    const getStragglerFactor = (n: number) => {
      const base = 1 + 0.05 * Math.log2(n)
      if (stragglerStrategy === 'threshold') return 1.0 // Clipped entirely
      if (stragglerStrategy === 'redundancy') return 1 + (base - 1) * 0.3 // Significantly reduced
      return base
    }
    
    let totalTimeSeconds = 0
    let mode = isMoE ? "MoE" : "Data Parallel (DiLoCo)"
    let globalCommSec = 0
    let regionalCommSec = 0
    let computeBlockSec = 0
    let latencyPenaltySec = 0

    if (!isSharded) {
      // --- Data Parallel / EP Mode ---
      const payloadBits = (parameters * bytesPerValue * 8) / compression
      
      if (useHierarchy) {
        mode = isMoE ? "Hierarchical MoE" : "Hierarchical DiLoCo"
        const numGroups = effectiveNodes / nodesPerGroup
        
        // Regional Sync (Every 'innerSteps')
        regionalCommSec = ((2 * payloadBits) / (regionalBandwidth * 1e6) + (regionalLatency / 1000)) * getStragglerFactor(nodesPerGroup)
        
        // Global Sync (Every 'totalInnerSteps') - Only leaders communicate
        globalCommSec = ((2 * payloadBits) / (bandwidthMbps * 1e6) + (latencyMs / 1000)) * getStragglerFactor(numGroups)
        
        // One global cycle = H_global regional cycles
        // Streaming overlaps the sync with the compute block
        const regionalCycleTime = streamingEnabled 
          ? Math.max(innerSteps * computeTimePerStep, regionalCommSec)
          : (innerSteps * computeTimePerStep) + regionalCommSec

        const globalCycleTime = streamingEnabled
          ? Math.max(regionalSteps * regionalCycleTime, globalCommSec)
          : (regionalSteps * regionalCycleTime) + globalCommSec
        
        const totalGlobalCycles = (tokens / (localBatch * effectiveNodes)) / (innerSteps * regionalSteps)
        totalTimeSeconds = totalGlobalCycles * globalCycleTime
        computeBlockSec = (innerSteps * regionalSteps) * computeTimePerStep
      } else {
        computeBlockSec = innerSteps * computeTimePerStep
        globalCommSec = ((2 * payloadBits) / (bandwidthMbps * 1e6) + (latencyMs / 1000)) * getStragglerFactor(effectiveNodes)
        
        const effectiveOuterTime = streamingEnabled
          ? Math.max(computeBlockSec, globalCommSec)
          : computeBlockSec + globalCommSec

        const totalOuterSteps = (tokens / (localBatch * effectiveNodes)) / innerSteps
        totalTimeSeconds = totalOuterSteps * effectiveOuterTime
      }
    } else {
      // --- PP-Group DiLoCo Mode ---
      // Group S nodes into PP clusters, run DiLoCo across N/S groups.
      // When hierarchy is enabled, PP groups use regional interconnect.
      const numGroups = Math.floor(effectiveNodes / ppStages)
      const hiddenDim = 0.03 * Math.sqrt(parameters)
      const activationBits = (localBatch * hiddenDim * bytesPerValue * 8) / ppCompression

      // PP intra-group uses regional interconnect if hierarchy enabled, else WAN
      const ppBandwidth = useHierarchy ? regionalBandwidth : bandwidthMbps
      const ppLatencyMs = useHierarchy ? regionalLatency : latencyMs

      const commPerMicroSec = (2 * activationBits / microBatches) / (ppBandwidth * 1e6)
      const computePerMicroSec = computeTimePerStep / microBatches
      const ppLatencySec = ppLatencyMs / 1000
      const ppStraggler = getStragglerFactor(ppStages)

      // GPipe bubble formula for one training step within a PP group
      const ppStepTime = (microBatches + ppStages - 1) * (computePerMicroSec + (commPerMicroSec + ppLatencySec) * ppStraggler)

      if (numGroups < 2) {
        // Not enough nodes to form multiple PP groups — pure PP over WAN
        mode = `PP over WAN (${ppStages} stages)` + (isMoE ? " + MoE" : "")
        globalCommSec = (microBatches + ppStages - 1) * commPerMicroSec * ppStraggler
        latencyPenaltySec = (microBatches + ppStages - 1) * ppLatencySec * ppStraggler
        computeBlockSec = (microBatches + ppStages - 1) * computePerMicroSec
        totalTimeSeconds = (tokens / (localBatch * Math.max(1, effectiveNodes / ppStages))) * ppStepTime
      } else {
        // PP-Group DiLoCo: DiLoCo outer loop across groups
        mode = `PP-Group DiLoCo (${ppStages}×${numGroups})` + (isMoE ? " + MoE" : "")
        const payloadBits = (parameters * bytesPerValue * 8) / compression
        computeBlockSec = innerSteps * ppStepTime
        globalCommSec = ((2 * payloadBits) / (bandwidthMbps * 1e6) + (latencyMs / 1000)) * getStragglerFactor(numGroups)

        const effectiveOuterTime = streamingEnabled
          ? Math.max(computeBlockSec, globalCommSec)
          : computeBlockSec + globalCommSec

        const totalOuterSteps = (tokens / (localBatch * numGroups)) / innerSteps
        totalTimeSeconds = totalOuterSteps * effectiveOuterTime
      }
    }

    // Effective compute time accounts for algorithmic penalty
    const totalTimeDays = totalTimeSeconds / (24 * 3600)
    const effectiveDays = totalTimeDays / algorithmicEfficiency
    const effectiveSeconds = effectiveDays * 24 * 3600

    // Global Utilization Metrics (End-to-End)
    // Theoretical FLOPs = 6 * parameters * tokens
    // Hardware Max FLOPs = numNodes * pflopsPerNode * 1e15 * effectiveSeconds
    const theoreticalFlops = 6 * computeParams * tokens
    const hardwareMaxFlops = numNodes * (pflopsPerNode * 1e15) * effectiveSeconds
    const globalMfu = (theoreticalFlops / hardwareMaxFlops)
    const globalHfu = globalMfu / 0.8 // MFU is ~80% of HFU per research

    // Calculate Dynamic Max Training Run (Epoch - The Longest Training Run)
    // Formula: L = 1 / (gH + gS + gI)
    const combinedGrowth = hwGrowth + swGrowth + investGrowth
    const computedMaxDays = (1 / (combinedGrowth * Math.LN10)) * 365
    const maxDays = manualMaxDays ? manualMaxDaysValue : computedMaxDays

    // Compute max model sizes that fit on one node at each precision
    const maxParamsFP16 = (vramPerNode * 1e9) / 16
    const maxParamsFP8 = (vramPerNode * 1e9) / 14
    const maxParamsFP4 = (vramPerNode * 1e9) / 13

    // PP-Group DiLoCo metadata
    const numPPGroups = isSharded ? Math.floor(effectiveNodes / ppStages) : 0
    const ppGroupMode = isSharded && numPPGroups >= 2
    const epReducedMemory = isMoE && expertParallelism !== 'none' && epDegree > 1
      && (parameters * bytesPerParam) > (vramPerNode * 1e9)  // would have been sharded without EP
      && !isSharded  // but EP made it fit

    setResults({
      mode,
      computeStepSec: computeTimePerStep.toFixed(2),
      globalCommSec: globalCommSec.toFixed(2),
      regionalCommSec: regionalCommSec.toFixed(2),
      latencySec: latencyPenaltySec.toFixed(2),
      computeBlockSec: computeBlockSec.toFixed(2),
      days: effectiveDays.toFixed(2),
      rawDays: totalTimeDays.toFixed(2),
      efficiency: (algorithmicEfficiency * 100).toFixed(1),
      globalMfu: (globalMfu * 100).toFixed(1),
      globalHfu: (globalHfu * 100).toFixed(1),
      isSharded,
      ppStages,
      numPPGroups,
      ppGroupMode,
      perNodeParams: (perNodeParams / 1e9).toFixed(1),
      epReducedMemory,
      maxParamsFP16: (maxParamsFP16 / 1e9).toFixed(0),
      maxParamsFP8: (maxParamsFP8 / 1e9).toFixed(0),
      maxParamsFP4: (maxParamsFP4 / 1e9).toFixed(0),
      hardwareFlops: hardwareMaxFlops,
      localEquivFlops: theoreticalFlops,
      maxDays: maxDays.toFixed(0),
      bottleneck: (globalCommSec + latencyPenaltySec) > computeBlockSec ? "Network" : "Compute",
      feasibility: effectiveDays < maxDays ? "Feasible" : `Impractical (>${maxDays.toFixed(0)} days)`
    })
  }

  const formatFlops = (flops: number) => {
    if (flops <= 0 || !isFinite(flops)) return '\u2014'
    const exp = Math.floor(Math.log10(flops))
    const mantissa = flops / Math.pow(10, exp)
    return `${mantissa.toFixed(2)} \u00d7 10^${exp}`
  }

  return (
    <div>
      <h1>Decentralized Training Simulator</h1>
      <p>Estimate feasibility of large-scale ML training over the internet.</p>

      <div className="simulator-container">
        <section>
          <h3>Model & Dataset</h3>
          <div className="input-group">
            <label>Model Type: <Tooltip text="Dense (all params active) or MoE (only experts active per token)." /></label>
            <select value={isMoE ? 'moe' : 'dense'} onChange={(e) => setIsMoE(e.target.value === 'moe')}>
              <option value="dense">Dense</option>
              <option value="moe">Mixture of Experts (MoE)</option>
            </select>
            <span>Type</span>
          </div>
          <div className="input-group">
            <label>Total Parameters (B): <Tooltip text="Total parameters in memory. For MoE, this includes all experts." /></label>
            <input 
              type="number" 
              min="1" 
              max="15000" 
              step="10" 
              value={parameters / 1e9} 
              onChange={(e) => setParameters(Number(e.target.value) * 1e9)} 
            />
            <span>B</span>
          </div>
          {isMoE && (
            <>
              <div className="input-group">
                <label>Active Params (B): <Tooltip text="Parameters active per token. Determines FLOPs/step." /></label>
                <input 
                  type="number" 
                  min="1" 
                  max={parameters / 1e9} 
                  step="1" 
                  value={activeParams / 1e9} 
                  onChange={(e) => setActiveParams(Number(e.target.value) * 1e9)} 
                />
                <span>B</span>
              </div>
              <div className="input-group">
                <label>MoE Layers: <Tooltip text="Number of layers that use expert routing. Each adds All-to-All latency." /></label>
                <input 
                  type="number" 
                  min="1" 
                  max="200" 
                  step="1" 
                  value={moeLayers} 
                  onChange={(e) => setMoeLayers(Number(e.target.value))} 
                />
                <span>Layers</span>
              </div>
              <div className="input-group">
                <label>Expert Parallelism: <Tooltip text="How experts are sharded. Global EP over WAN is extremely slow." /></label>
                <select value={expertParallelism} onChange={(e) => setExpertParallelism(e.target.value)}>
                  <option value="none">None (Replicated)</option>
                  <option value="regional">Regional (Low Latency)</option>
                  <option value="global">Global (WAN All-to-All)</option>
                </select>
                <span>Scope</span>
              </div>
            </>
          )}
          <div className="input-group">
            <label>Tokens (T): <Tooltip text="Total number of training tokens in trillions." /></label>
            <input
              type="number"
              min="0.1"
              max="100"
              step="0.1"
              value={tokens / 1e12}
              onChange={(e) => setTokens(Number(e.target.value) * 1e12)}
            />
            <span>T</span>
          </div>
          <div className="input-group">
            <label>Local Batch (tokens): <Tooltip text="Tokens per training step per node. Equal to num_sequences × seq_length (e.g. 32 × 4096 = 131,072). Determines compute per step and activation size in PP mode." /></label>
            <input
              type="number"
              min="1024"
              max="4194304"
              step="1024"
              value={localBatch}
              onChange={(e) => setLocalBatch(Number(e.target.value))}
            />
            <span>tokens</span>
          </div>
        </section>

        <section>
          <h3>Infrastructure (Global)</h3>
          <div className="input-group">
            <label>Number of Nodes: <Tooltip text="Total number of geographically distributed training replicas." /></label>
            <input 
              type="number" 
              min="1" 
              max="5000" 
              step="1" 
              value={numNodes} 
              onChange={(e) => setNumNodes(Number(e.target.value))} 
            />
            <span>Nodes</span>
          </div>
          <div className="input-group">
            <label>WAN Bandwidth (Mbps): <Tooltip text="Inter-node upload/download speed over the internet." /></label>
            <input 
              type="number" 
              min="1" 
              max="10000" 
              step="10" 
              value={bandwidthMbps} 
              onChange={(e) => setBandwidthMbps(Number(e.target.value))} 
            />
            <span>Mbps</span>
          </div>
          <div className="input-group">
            <label>WAN Latency (ms): <Tooltip text="Average round-trip time between nodes." /></label>
            <input 
              type="number" 
              min="1" 
              max="1000" 
              step="10" 
              value={latencyMs} 
              onChange={(e) => setLatencyMs(Number(e.target.value))} 
            />
            <span>ms</span>
          </div>
          <div className="input-group">
            <label>Node VRAM (GB): <Tooltip text="Total HBM memory available per node across all its GPUs." /></label>
            <input 
              type="number" 
              min="8" 
              max="5000" 
              step="8" 
              value={vramPerNode} 
              onChange={(e) => setVramPerNode(Number(e.target.value))} 
            />
            <span>GB</span>
          </div>
          <div className="input-group">
            <label>Node PFLOPS: <Tooltip text="Peak theoretical FP16/BF16 performance per node." /></label>
            <input 
              type="number" 
              min="1" 
              max="1000" 
              step="1" 
              value={pflopsPerNode} 
              onChange={(e) => setPflopsPerNode(Number(e.target.value))} 
            />
            <span>PFLOPS</span>
          </div>
          <div className="input-group">
            <label>Base MFU (%): <Tooltip text="Standard Model FLOPs Utilization for this hardware on local training." /></label>
            <input 
              type="number" 
              min="5" 
              max="80" 
              step="1" 
              value={mfu * 100} 
              onChange={(e) => setMfu(Number(e.target.value) / 100)} 
            />
            <span>%</span>
          </div>
          {mfu > 0.6 && (
            <p style={{ color: '#ffcc00', fontSize: '0.8em', marginTop: '5px' }}>
              ⚠️ MFU &gt; 60% is rare in practice. Most large-scale jobs peak at 40-50%.
            </p>
          )}
        </section>

        <section>
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            <h3 style={{ margin: 0 }}>Hierarchical Sync</h3>
            <input type="checkbox" checked={useHierarchy} onChange={(e) => setUseHierarchy(e.target.checked)} />
            <Tooltip text="Enables regional clusters to sync locally before global WAN synchronization." />
          </div>
          {useHierarchy && (
            <div style={{ borderLeft: '2px solid #38bdf8', paddingLeft: '15px', marginTop: '1.5rem' }}>
              <div className="input-group">
                <label>Group Nodes: <Tooltip text="Number of nodes in each local/regional group." /></label>
                <input 
                  type="number" 
                  min="2" 
                  max={numNodes} 
                  step="1" 
                  value={nodesPerGroup} 
                  onChange={(e) => setNodesPerGroup(Number(e.target.value))} 
                />
                <span>Nodes</span>
              </div>
              <div className="input-group">
                <label>Regional BW (Mbps): <Tooltip text="High-speed bandwidth within the regional group (e.g., LAN/Metro)." /></label>
                <input 
                  type="number" 
                  min="100" 
                  max="100000" 
                  step="100" 
                  value={regionalBandwidth} 
                  onChange={(e) => setRegionalBandwidth(Number(e.target.value))} 
                />
                <span>Mbps</span>
              </div>
              <div className="input-group">
                <label>Regional Latency (ms): <Tooltip text="Lower latency within the regional cluster." /></label>
                <input 
                  type="number" 
                  min="1" 
                  max="100" 
                  step="1" 
                  value={regionalLatency} 
                  onChange={(e) => setRegionalLatency(Number(e.target.value))} 
                />
                <span>ms</span>
              </div>
              <div className="input-group">
                <label>Regional Sync Steps: <Tooltip text="Number of regional sync cycles before one global WAN sync." /></label>
                <input 
                  type="number" 
                  min="1" 
                  max="100" 
                  step="1" 
                  value={regionalSteps} 
                  onChange={(e) => setRegionalSteps(Number(e.target.value))} 
                />
                <span>Steps</span>
              </div>
            </div>
          )}
        </section>

        <section>
          <h3>Algorithm Settings</h3>
          <div className="input-group">
            <label>Precision: <Tooltip text="The numeric format used for training. Lower precision reduces bandwidth but can impact stability." /></label>
            <select value={precision} onChange={(e) => setPrecision(e.target.value)}>
              <option value="FP16">FP16 / BF16 (2 bytes)</option>
              <option value="FP8">FP8 (1 byte)</option>
              <option value="FP4">FP4 (0.5 byte)</option>
            </select>
            <span>Precision</span>
          </div>
          <div className="input-group">
            <label>Streaming DiLoCo: <Tooltip text="Overlap synchronization with the next compute block. Hides network latency if compute time > sync time." /></label>
            <input 
              type="checkbox" 
              checked={streamingEnabled} 
              onChange={(e) => setStreamingEnabled(e.target.checked)} 
            />
            <span>Enabled</span>
          </div>
          <div className="input-group">
            <label>Inner Steps (Local): <Tooltip text="Steps performed locally between synchronizations." /></label>
            <input 
              type="number" 
              min="1" 
              max="1000" 
              step="1" 
              value={innerSteps} 
              onChange={(e) => setInnerSteps(Number(e.target.value))} 
            />
            <span>Steps</span>
          </div>
          <div className="input-group">
            <label>Weight Compression: <Tooltip text="Quantization and sparsification factor for weight synchronization." /></label>
            <input 
              type="number" 
              min="1" 
              max="100" 
              step="1" 
              value={compression} 
              onChange={(e) => setCompression(Number(e.target.value))} 
            />
            <span>x</span>
          </div>
          <div className="input-group">
            <label>Activation Compression: <Tooltip text="Compression factor for inter-node activations in Pipeline Parallel mode." /></label>
            <input 
              type="number" 
              min="1" 
              max="100" 
              step="1" 
              value={ppCompression} 
              onChange={(e) => setPpCompression(Number(e.target.value))} 
            />
            <span>x</span>
          </div>
          <div className="input-group">
            <label>Straggler Mitigation: <Tooltip text="Strategies to prevent slow nodes from delaying the entire cluster." /></label>
            <select 
              value={stragglerStrategy} 
              onChange={(e) => setStragglerStrategy(e.target.value)} 
            >
              <option value="none">None (Blocking)</option>
              <option value="threshold">Threshold (90% cut-off)</option>
              <option value="redundancy">Backup Workers (10% extra)</option>
            </select>
            <span>Strategy</span>
          </div>
        </section>

        <section>
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center', cursor: 'pointer' }} onClick={() => setShowMaxDuration(!showMaxDuration)}>
            <h3 style={{ margin: 0 }}>Maximum Training Duration {showMaxDuration ? '▼' : '▶'}</h3>
          </div>
          <p style={{ fontSize: '0.8em', color: '#64748b', marginTop: '4px' }}>
            Based on Epoch's{' '}
            <a href="https://epoch.ai/blog/the-longest-training-run" target="_blank" rel="noopener noreferrer" style={{ color: '#38bdf8' }}>The Longest Training Run</a>.
            See their{' '}
            <a href="https://epoch.ai/data/trends" target="_blank" rel="noopener noreferrer" style={{ color: '#38bdf8' }}>Trends dashboard</a>{' '}
            for the latest growth rate estimates.
          </p>
          {showMaxDuration && (
            <div style={{ borderLeft: '2px solid #38bdf8', paddingLeft: '15px', marginBottom: '20px', marginTop: '1.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
                <input type="checkbox" checked={manualMaxDays} onChange={(e) => setManualMaxDays(e.target.checked)} id="manualOverride" />
                <label htmlFor="manualOverride" style={{ fontSize: '0.85em', color: '#94a3b8' }}>Manual override</label>
                {manualMaxDays && (
                  <input
                    type="number"
                    min="1"
                    max="3650"
                    step="1"
                    value={manualMaxDaysValue}
                    onChange={(e) => setManualMaxDaysValue(Number(e.target.value))}
                    style={{ background: '#2a2a2a', color: '#e2e8f0', border: '1px solid #475569', padding: '6px 10px', borderRadius: '6px', width: '80px', fontSize: '0.9em', outline: 'none' }}
                  />
                )}
                {manualMaxDays && <span style={{ fontSize: '0.85em', color: '#64748b' }}>days</span>}
              </div>

              {!manualMaxDays && (
                <>
                  <div style={{ display: 'flex', gap: '4px', marginBottom: '16px' }}>
                    {(['percent', 'oom', 'multiple'] as const).map((u) => (
                      <label key={u} style={{
                        display: 'flex', alignItems: 'center', gap: '4px', padding: '4px 10px',
                        borderRadius: '6px', fontSize: '0.8em', cursor: 'pointer',
                        background: growthUnit === u ? '#1e3a5f' : 'transparent',
                        border: growthUnit === u ? '1px solid #38bdf8' : '1px solid #475569',
                        color: growthUnit === u ? '#38bdf8' : '#94a3b8'
                      }}>
                        <input
                          type="radio"
                          name="growthUnit"
                          value={u}
                          checked={growthUnit === u}
                          onChange={() => setGrowthUnit(u)}
                          style={{ display: 'none' }}
                        />
                        {u === 'percent' ? '%/yr' : u === 'oom' ? 'OOM/yr' : '\u00d7/yr'}
                      </label>
                    ))}
                  </div>
                  <GrowthInput
                    label="HW Growth"
                    value={hwGrowth}
                    onChange={setHwGrowth}
                    tooltip="Rate of hardware FLOPs/dollar improvement."
                    unit={growthUnit}
                  />
                  <GrowthInput
                    label="SW Growth"
                    value={swGrowth}
                    onChange={setSwGrowth}
                    tooltip="Rate of algorithmic efficiency / compute-saving software improvements."
                    unit={growthUnit}
                  />
                  <GrowthInput
                    label="Invest Growth"
                    value={investGrowth}
                    onChange={setInvestGrowth}
                    tooltip="Rate of increase in total capital investment for training runs."
                    unit={growthUnit}
                  />
                  <p style={{ fontSize: '0.75em', color: '#64748b', marginTop: '15px' }}>
                    High growth rates imply shorter optimal runs — delaying for better tech/budget becomes more attractive.
                  </p>
                </>
              )}

              {results && (
                <div style={{ marginTop: '12px', padding: '12px', background: '#1e293b', borderRadius: '8px', border: '1px solid #334155' }}>
                  <p style={{ margin: 0, fontSize: '0.85em', color: '#94a3b8' }}>
                    Maximum practical training run
                  </p>
                  <p style={{ margin: '4px 0 0 0', fontSize: '1.3em', fontWeight: 700, color: '#38bdf8' }}>
                    {results.maxDays} days
                    <span style={{ fontSize: '0.6em', fontWeight: 400, color: '#64748b', marginLeft: '8px' }}>
                      ({(Number(results.maxDays) / 30.44).toFixed(1)} months)
                    </span>
                  </p>
                </div>
              )}
            </div>
          )}
        </section>

        {results && (
          <div className="results-card" style={{ borderTop: results.isSharded ? '4px solid #38bdf8' : '1px solid #334155' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
              <h2 style={{ margin: 0 }}>Results</h2>
              <span style={{ background: '#38bdf8', color: '#0f172a', padding: '4px 12px', borderRadius: '20px', fontSize: '0.75em', fontWeight: 700, textTransform: 'uppercase' }}>
                {results.mode}
              </span>
            </div>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' }}>
              <div>
                <p style={{ color: '#94a3b8', margin: '0 0 4px 0', fontSize: '0.9em' }}>Total Training Time</p>
                <h1 style={{ color: results.days < 365 ? '#10b981' : '#f43f5e', margin: 0 }}>{results.days} Days</h1>
                <p style={{ fontWeight: 600, marginTop: '8px' }}>{results.feasibility}</p>
                <div style={{ fontSize: '0.85em', color: '#64748b', marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <span>{results.efficiency}% algorithmic efficiency</span>
                  <span>Global MFU: <span style={{ color: results.globalMfu < 20 ? '#f43f5e' : '#94a3b8' }}>{results.globalMfu}%</span> | HFU: {results.globalHfu}%</span>
                </div>
                {results.globalMfu < 20 && (
                  <p style={{ fontSize: '0.75em', color: '#f43f5e', marginTop: '12px' }}>
                    * Inefficient run: Communication overhead or algorithmic penalty is dominating.
                  </p>
                )}
              </div>
              
              <div style={{ background: '#1e293b', padding: '15px', borderRadius: '10px', border: '1px solid #334155' }}>
                <p style={{ color: '#94a3b8', margin: '0 0 8px 0', fontSize: '0.85em', fontWeight: 600 }}>Bottleneck: {results.bottleneck}</p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '0.85em', color: '#cbd5e1' }}>
                  <p style={{ margin: 0 }}>Global Sync: {results.globalCommSec}s</p>
                  {useHierarchy && <p style={{ margin: 0 }}>Regional Sync: {results.regionalCommSec}s</p>}
                  <p style={{ margin: 0 }}>Compute Block: {results.computeBlockSec}s</p>
                </div>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginTop: '20px' }}>
              <div style={{ background: '#1e293b', padding: '15px', borderRadius: '10px', border: '1px solid #334155' }}>
                <p style={{ color: '#94a3b8', margin: '0 0 4px 0', fontSize: '0.75em', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Hardware FLOPs Performed
                  <Tooltip text="Total FLOPs burned by all hardware over the training run, including idle time, pipeline bubbles, straggler waits, and activation recomputation." />
                </p>
                <p style={{ margin: 0, fontSize: '1.1em', fontWeight: 600, color: '#e2e8f0' }}>
                  {formatFlops(results.hardwareFlops)} FLOP
                </p>
              </div>
              <div style={{ background: '#1e293b', padding: '15px', borderRadius: '10px', border: '1px solid #334155' }}>
                <p style={{ color: '#94a3b8', margin: '0 0 4px 0', fontSize: '0.75em', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Local-Equivalent Compute
                  <Tooltip text="Compute needed to train the same model to the same performance on a single ideal cluster with no distributed training overhead (no DiLoCo penalty, no compression loss, no straggler waste, no pipeline bubbles)." />
                </p>
                <p style={{ margin: 0, fontSize: '1.1em', fontWeight: 600, color: '#e2e8f0' }}>
                  {formatFlops(results.localEquivFlops)} FLOP
                </p>
              </div>
            </div>

            {results.epReducedMemory && (
              <div style={{ marginTop: '20px', padding: '15px', background: '#0f2318', border: '1px solid #166534', borderRadius: '10px' }}>
                <p style={{ color: '#86efac', margin: '0 0 8px 0', fontSize: '0.85em', fontWeight: 700 }}>
                  Expert Parallelism reduced per-node memory to {results.perNodeParams}B params
                </p>
                <p style={{ color: '#94a3b8', margin: 0, fontSize: '0.8em' }}>
                  Without EP, the full {(parameters / 1e9).toFixed(0)}B model would require pipeline parallelism.
                  EP shards experts across nodes, so each node stores only shared parameters + its local experts,
                  allowing DiLoCo mode instead of PP.
                </p>
              </div>
            )}

            {results.isSharded && results.ppGroupMode && (
              <div style={{ marginTop: '20px', padding: '15px', background: '#1c1a17', border: '1px solid #78630d', borderRadius: '10px' }}>
                <p style={{ color: '#fde68a', margin: '0 0 8px 0', fontSize: '0.85em', fontWeight: 700 }}>
                  PP-Group DiLoCo: {results.ppStages} PP stages × {results.numPPGroups} DiLoCo groups ({results.perNodeParams}B params/node)
                </p>
                <p style={{ color: '#94a3b8', margin: '0 0 12px 0', fontSize: '0.8em' }}>
                  The model exceeds single-node VRAM. Nodes are grouped into {results.ppStages}-node PP clusters that shard the model,
                  with DiLoCo synchronization across {results.numPPGroups} groups every {innerSteps} steps.
                  {useHierarchy
                    ? ' PP handoffs use regional interconnect. DiLoCo sync uses WAN.'
                    : ' Both PP handoffs and DiLoCo sync use WAN.'}
                </p>
                <div style={{ fontSize: '0.8em', color: '#cbd5e1', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  <p style={{ margin: 0 }}>
                    <strong style={{ color: '#fbbf24' }}>To improve performance:</strong> Max model on one node — FP16: {results.maxParamsFP16}B, FP8: {results.maxParamsFP8}B, FP4: {results.maxParamsFP4}B
                  </p>
                  {isMoE && expertParallelism === 'none' && (
                    <p style={{ margin: 0 }}>
                      <strong style={{ color: '#fbbf24' }}>Enable Expert Parallelism:</strong> Shard experts across nodes to reduce per-node memory and potentially avoid PP entirely
                    </p>
                  )}
                </div>
              </div>
            )}

            {results.isSharded && !results.ppGroupMode && (
              <div style={{ marginTop: '20px', padding: '15px', background: '#1c1317', border: '1px solid #7f1d1d', borderRadius: '10px' }}>
                <p style={{ color: '#fca5a5', margin: '0 0 8px 0', fontSize: '0.85em', fontWeight: 700 }}>
                  Insufficient nodes for PP-Group DiLoCo ({results.ppStages} stages needed, only {numNodes} nodes)
                </p>
                <p style={{ color: '#94a3b8', margin: '0 0 12px 0', fontSize: '0.8em' }}>
                  The model requires {results.ppStages} PP stages but there aren't enough nodes to form multiple groups.
                  All nodes form a single pipeline, with no DiLoCo parallelism. This is extremely slow over WAN.
                </p>
                <div style={{ fontSize: '0.8em', color: '#cbd5e1', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  <p style={{ margin: 0 }}>
                    <strong style={{ color: '#fbbf24' }}>Reduce precision:</strong> Max model on one node — FP16: {results.maxParamsFP16}B, FP8: {results.maxParamsFP8}B, FP4: {results.maxParamsFP4}B
                  </p>
                  <p style={{ margin: 0 }}>
                    <strong style={{ color: '#fbbf24' }}>Train a smaller model:</strong> Stay under {results.maxParamsFP4}B params (FP4) to use DiLoCo
                  </p>
                  {isMoE && expertParallelism === 'none' && (
                    <p style={{ margin: 0 }}>
                      <strong style={{ color: '#fbbf24' }}>Enable Expert Parallelism:</strong> Shard experts across nodes to reduce per-node memory
                    </p>
                  )}
                  <p style={{ margin: 0 }}>
                    <strong style={{ color: '#fbbf24' }}>Add more nodes:</strong> Need at least {results.ppStages * 2} nodes for PP-Group DiLoCo
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
      
      <p className="read-the-docs">
        Based on the MIRI Technical Governance Team project.
      </p>
    </div>
  )
}

export default App
