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
  const [mfu, setMfu] = useState(0.4)

  // Algorithm Parameters
  const [innerSteps, setInnerSteps] = useState(128)
  const [compression, setCompression] = useState(16)
  const [localBatch, setLocalBatch] = useState(131072)
  const [ppCompression, setPpCompression] = useState(10) // Activations are harder to compress

  const [results, setResults] = useState<any>(null)

  useEffect(() => {
    calculate()
  }, [parameters, tokens, numNodes, pflopsPerNode, vramPerNode, bandwidthMbps, mfu, innerSteps, compression, localBatch, ppCompression])

  const calculate = () => {
    // 1. Memory Analysis
    // Standard training (Adam) takes ~16 bytes per param (Weights, Grads, Opt States)
    // ZeRO-3 / FSDP can reduce this to ~2-4 bytes per param + sharded overhead
    const bytesPerParam = 12 // Conservative middle ground with some optimization
    const totalMemoryRequired = parameters * bytesPerParam
    const totalMemoryAvailable = numNodes * vramPerNode * 1e9
    
    const isSharded = (parameters * bytesPerParam) > (vramPerNode * 1e9)
    const ppStages = Math.ceil(totalMemoryRequired / (vramPerNode * 1e9))
    
    // 2. Compute Time (Same for both)
    const flopsPerStep = 6 * parameters * localBatch
    const nodeComputePower = pflopsPerNode * 1e15 * mfu
    const computeTimePerStep = flopsPerStep / nodeComputePower
    
    let totalTimeSeconds = 0
    let mode = "Data Parallel (DiLoCo)"
    let commTimeSec = 0
    let computeBlockSec = 0

    if (!isSharded) {
      // --- DiLoCo Mode ---
      const payloadBits = (parameters * 2 * 8) / compression // Weights sync
      commTimeSec = (2 * payloadBits) / (bandwidthMbps * 1e6)
      computeBlockSec = innerSteps * computeTimePerStep
      
      const effectiveOuterTime = Math.max(computeBlockSec, commTimeSec)
      const totalOuterSteps = (tokens / (localBatch * numNodes)) / innerSteps
      totalTimeSeconds = totalOuterSteps * effectiveOuterTime
    } else {
      // --- Pipeline Parallel Mode (SWARM/Protocol style) ---
      mode = `Pipeline Parallel (${ppStages} stages)`
      // Activations must be sent for EVERY step (or micro-batch)
      // Activation Size ~= Batch * Hidden_Dim * Precision
      // Heuristic for Hidden Dim: 0.004 * sqrt(Params)
      const hiddenDim = 0.004 * Math.sqrt(parameters)
      const activationBits = (localBatch * hiddenDim * 2 * 8) / ppCompression
      
      // Each step must send activations forward and gradients backward
      commTimeSec = (2 * activationBits) / (bandwidthMbps * 1e6)
      computeBlockSec = computeTimePerStep
      
      // Pipeline Parallel is extremely sensitive to latency
      // We assume activations/grads are blocking per step
      const timePerStep = computeBlockSec + commTimeSec
      totalTimeSeconds = (tokens / (localBatch * numNodes)) * timePerStep
    }

    const days = totalTimeSeconds / (24 * 3600)

    setResults({
      mode,
      computeStepSec: computeTimePerStep.toFixed(2),
      commSec: commTimeSec.toFixed(2),
      computeBlockSec: computeBlockSec.toFixed(2),
      days: days.toFixed(2),
      isSharded,
      bottleneck: commTimeSec > computeBlockSec ? "Network Bandwidth" : "Compute",
      feasibility: days < 365 ? "Feasible" : "Impractical (>1 year)"
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
            <label>Parameters (B): {parameters / 1e9}</label>
            <input type="range" min="1" max="15000" step="10" value={parameters / 1e9} onChange={(e) => setParameters(Number(e.target.value) * 1e9)} />
          </div>
          <div className="input-group">
            <label>Tokens (T): {tokens / 1e12}</label>
            <input type="range" min="0.1" max="100" step="0.1" value={tokens / 1e12} onChange={(e) => setTokens(Number(e.target.value) * 1e12)} />
          </div>
        </section>

        <section>
          <h3>Infrastructure</h3>
          <div className="input-group">
            <label>Number of Nodes: {numNodes}</label>
            <input type="range" min="1" max="5000" step="1" value={numNodes} onChange={(e) => setNumNodes(Number(e.target.value))} />
          </div>
          <div className="input-group">
            <label>Node VRAM (GB): {vramPerNode}</label>
            <input type="range" min="8" max="5000" step="8" value={vramPerNode} onChange={(e) => setVramPerNode(Number(e.target.value))} />
          </div>
          <div className="input-group">
            <label>Node PFLOPS: {pflopsPerNode}</label>
            <input type="range" min="1" max="1000" step="1" value={pflopsPerNode} onChange={(e) => setPflopsPerNode(Number(e.target.value))} />
          </div>
          <div className="input-group">
            <label>WAN Bandwidth (Mbps): {bandwidthMbps}</label>
            <input type="range" min="1" max="10000" step="10" value={bandwidthMbps} onChange={(e) => setBandwidthMbps(Number(e.target.value))} />
          </div>
        </section>

        <section>
          <h3>Algorithm Settings</h3>
          <div className="input-group">
            <label>DiLoCo Inner Steps: {innerSteps}</label>
            <input type="range" min="1" max="1000" step="1" value={innerSteps} onChange={(e) => setInnerSteps(Number(e.target.value))} />
          </div>
          <div className="input-group">
            <label>Weight Compression: {compression}x</label>
            <input type="range" min="1" max="100" step="1" value={compression} onChange={(e) => setCompression(Number(e.target.value))} />
          </div>
          <div className="input-group">
            <label>Activation Compression: {ppCompression}x</label>
            <input type="range" min="1" max="100" step="1" value={ppCompression} onChange={(e) => setPpCompression(Number(e.target.value))} />
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
              </div>
              <div>
                <p><strong>Bottleneck:</strong> {results.bottleneck}</p>
                {results.isSharded ? (
                  <p>Inter-stage Comm: {results.commSec}s / step</p>
                ) : (
                  <p>Weight Sync: {results.commSec}s / outer step</p>
                )}
                <p>Compute: {results.computeStepSec}s / step</p>
              </div>
            </div>
            
            {results.isSharded && (
              <div style={{ marginTop: '10px', fontSize: '0.9em', color: '#aaa' }}>
                * Model exceeds single-node VRAM. Simulator is assuming <strong>Pipeline Parallelism</strong> 
                (e.g., SWARM or Protocol Models). This mode is highly sensitive to per-step network latency.
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
