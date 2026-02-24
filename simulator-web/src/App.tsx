import { useState, useEffect } from 'react'

function App() {
  // Model Parameters
  const [parameters, setParameters] = useState(144e9) // 144B
  const [tokens, setTokens] = useState(12e12) // 12T
  
  // Hardware Parameters
  const [numNodes, setNumNodes] = useState(72)
  const [pflopsPerNode, setPflopsPerNode] = useState(32) // GH200 x16
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

  const [results, setResults] = useState<any>(null)

  useEffect(() => {
    calculate()
  }, [parameters, tokens, numNodes, pflopsPerNode, vramPerNode, bandwidthMbps, latencyMs, mfu, innerSteps, compression, localBatch, ppCompression, microBatches, useHierarchy, nodesPerGroup, regionalBandwidth, regionalLatency, regionalSteps])

  const calculate = () => {
    // 1. Memory Analysis
    const bytesPerParam = 12 
    const isSharded = (parameters * bytesPerParam) > (vramPerNode * 1e9)
    const ppStages = Math.ceil((parameters * bytesPerParam) / (vramPerNode * 1e9))
    
    // 2. Compute Time
    const flopsPerStep = 6 * parameters * localBatch
    const nodeComputePower = pflopsPerNode * 1e15 * mfu
    const computeTimePerStep = flopsPerStep / nodeComputePower
    
    // 3. Algorithmic Efficiency Penalty
    // Heuristic: loss of effective compute scales with total inner steps between global syncs
    const totalInnerSteps = useHierarchy ? innerSteps * regionalSteps : innerSteps
    // Alpha reduces slightly for larger models (more robust)
    const alpha = 0.08 * (1 / (1 + Math.log10(parameters / 1e9) / 5))
    const algorithmicEfficiency = Math.max(0.5, 1 - alpha * Math.log10(totalInnerSteps))
    
    let totalTimeSeconds = 0
    let mode = "Data Parallel (DiLoCo)"
    let globalCommSec = 0
    let regionalCommSec = 0
    let computeBlockSec = 0
    let latencyPenaltySec = 0

    if (!isSharded) {
      // --- Data Parallel Mode ---
      const payloadBits = (parameters * 2 * 8) / compression
      
      // Global Sync (Every 'totalInnerSteps')
      globalCommSec = (2 * payloadBits) / (bandwidthMbps * 1e6) + (latencyMs / 1000)
      
      if (useHierarchy) {
        mode = "Hierarchical DiLoCo"
        // Regional Sync (Every 'innerSteps')
        regionalCommSec = (2 * payloadBits) / (regionalBandwidth * 1e6) + (regionalLatency / 1000)
        
        // One global cycle = H_global regional cycles
        // One regional cycle = H_regional compute blocks
        const regionalCycleTime = Math.max(innerSteps * computeTimePerStep, regionalCommSec)
        const globalCycleTime = Math.max(regionalSteps * regionalCycleTime, globalCommSec)
        
        const totalGlobalCycles = (tokens / (localBatch * numNodes)) / (innerSteps * regionalSteps)
        totalTimeSeconds = totalGlobalCycles * globalCycleTime
        computeBlockSec = totalInnerSteps * computeTimePerStep
      } else {
        computeBlockSec = innerSteps * computeTimePerStep
        const effectiveOuterTime = Math.max(computeBlockSec, globalCommSec)
        const totalOuterSteps = (tokens / (localBatch * numNodes)) / innerSteps
        totalTimeSeconds = totalOuterSteps * effectiveOuterTime
      }
    } else {
      // --- Pipeline Parallel Mode ---
      mode = `Pipeline Parallel (${ppStages} stages)`
      const hiddenDim = 0.004 * Math.sqrt(parameters)
      const activationBits = (localBatch * hiddenDim * 2 * 8) / ppCompression
      
      const commPerMicroSec = (2 * activationBits / microBatches) / (bandwidthMbps * 1e6)
      const computePerMicroSec = computeTimePerStep / microBatches
      const latencySec = latencyMs / 1000

      const timePerStep = (microBatches + ppStages - 1) * (computePerMicroSec + commPerMicroSec + latencySec)
      
      globalCommSec = (microBatches + ppStages - 1) * commPerMicroSec
      latencyPenaltySec = (microBatches + ppStages - 1) * latencySec
      computeBlockSec = (microBatches + ppStages - 1) * computePerMicroSec
      
      totalTimeSeconds = (tokens / (localBatch * (numNodes / ppStages))) * timePerStep
    }

    // Effective compute time accounts for algorithmic penalty
    const totalTimeDays = totalTimeSeconds / (24 * 3600)
    const effectiveDays = totalTimeDays / algorithmicEfficiency

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
      isSharded,
      bottleneck: (globalCommSec + latencyPenaltySec) > computeBlockSec ? "Network" : "Compute",
      feasibility: effectiveDays < 365 ? "Feasible" : "Impractical (>1 year)"
    })
  }

  return (
    <div>
      <h1>Decentralized Training Simulator</h1>
      <p>Estimate feasibility of large-scale ML training over the internet.</p>

      <div className="simulator-container">
        <section>
          <h3>Model & Dataset</h3>
          <div className="input-group">
            <label>Parameters (B):</label>
            <input type="range" min="1" max="15000" step="10" value={parameters / 1e9} onChange={(e) => setParameters(Number(e.target.value) * 1e9)} />
            <span>{parameters / 1e9}</span>
          </div>
          <div className="input-group">
            <label>Tokens (T):</label>
            <input type="range" min="0.1" max="100" step="0.1" value={tokens / 1e12} onChange={(e) => setTokens(Number(e.target.value) * 1e12)} />
            <span>{tokens / 1e12}</span>
          </div>
        </section>

        <section>
          <h3>Infrastructure (Global)</h3>
          <div className="input-group">
            <label>Number of Nodes:</label>
            <input type="range" min="1" max="5000" step="1" value={numNodes} onChange={(e) => setNumNodes(Number(e.target.value))} />
            <span>{numNodes}</span>
          </div>
          <div className="input-group">
            <label>WAN Bandwidth (Mbps):</label>
            <input type="range" min="1" max="10000" step="10" value={bandwidthMbps} onChange={(e) => setBandwidthMbps(Number(e.target.value))} />
            <span>{bandwidthMbps}</span>
          </div>
          <div className="input-group">
            <label>WAN Latency (ms):</label>
            <input type="range" min="1" max="1000" step="10" value={latencyMs} onChange={(e) => setLatencyMs(Number(e.target.value))} />
            <span>{latencyMs}</span>
          </div>
          <div className="input-group">
            <label>Node VRAM (GB):</label>
            <input type="range" min="8" max="5000" step="8" value={vramPerNode} onChange={(e) => setVramPerNode(Number(e.target.value))} />
            <span>{vramPerNode}</span>
          </div>
          <div className="input-group">
            <label>Node PFLOPS:</label>
            <input type="range" min="1" max="1000" step="1" value={pflopsPerNode} onChange={(e) => setPflopsPerNode(Number(e.target.value))} />
            <span>{pflopsPerNode}</span>
          </div>
        </section>

        <section>
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            <h3>Hierarchical Sync</h3>
            <input type="checkbox" checked={useHierarchy} onChange={(e) => setUseHierarchy(e.target.checked)} />
          </div>
          {useHierarchy && (
            <div style={{ borderLeft: '2px solid #646cff', paddingLeft: '15px' }}>
              <div className="input-group">
                <label>Regional Bandwidth (Mbps):</label>
                <input type="range" min="100" max="100000" step="100" value={regionalBandwidth} onChange={(e) => setRegionalBandwidth(Number(e.target.value))} />
                <span>{regionalBandwidth}</span>
              </div>
              <div className="input-group">
                <label>Regional Latency (ms):</label>
                <input type="range" min="1" max="100" step="1" value={regionalLatency} onChange={(e) => setRegionalLatency(Number(e.target.value))} />
                <span>{regionalLatency}</span>
              </div>
              <div className="input-group">
                <label>Regional Sync Steps:</label>
                <input type="range" min="1" max="100" step="1" value={regionalSteps} onChange={(e) => setRegionalSteps(Number(e.target.value))} />
                <span>{regionalSteps}</span>
              </div>
            </div>
          )}
        </section>

        <section>
          <h3>Algorithm Settings</h3>
          <div className="input-group">
            <label>Inner Steps (Local):</label>
            <input type="range" min="1" max="1000" step="1" value={innerSteps} onChange={(e) => setInnerSteps(Number(e.target.value))} />
            <span>{innerSteps}</span>
          </div>
          <div className="input-group">
            <label>Weight Compression:</label>
            <input type="range" min="1" max="100" step="1" value={compression} onChange={(e) => setCompression(Number(e.target.value))} />
            <span>{compression}x</span>
          </div>
          <div className="input-group">
            <label>Activation Compression:</label>
            <input type="range" min="1" max="100" step="1" value={ppCompression} onChange={(e) => setPpCompression(Number(e.target.value))} />
            <span>{ppCompression}x</span>
          </div>
        </section>

        {results && (
          <div className="results-card" style={{ border: results.isSharded ? '2px solid #646cff' : 'none' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h2>Results</h2>
              <span style={{ background: '#646cff', padding: '4px 12px', borderRadius: '20px', fontSize: '0.8em' }}>
                Mode: {results.mode}
              </span>
            </div>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
              <div>
                <p><strong>Total Training Time:</strong></p>
                <h1 style={{ color: results.days < 365 ? '#4caf50' : '#ff4444', margin: 0 }}>{results.days} Days</h1>
                <p>{results.feasibility}</p>
                <p style={{ fontSize: '0.8em', color: '#aaa' }}>Includes {results.efficiency}% algorithmic efficiency</p>
              </div>
              <div>
                <p><strong>Bottleneck:</strong> {results.bottleneck}</p>
                <p>Global Sync: {results.globalCommSec}s</p>
                {useHierarchy && <p>Regional Sync: {results.regionalCommSec}s</p>}
                <p>Compute Block: {results.computeBlockSec}s</p>
              </div>
            </div>
            
            {results.isSharded && (
              <div style={{ marginTop: '10px', fontSize: '0.9em', color: '#aaa', maxWidth: '100%', overflowWrap: 'break-word' }}>
                * Model exceeds single-node VRAM. Pipeline Parallelism is active. 
                Network latency adds idle time to every micro-batch handover.
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
