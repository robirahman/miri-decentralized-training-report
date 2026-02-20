import { useState, useEffect } from 'react'

function App() {
  // Model Parameters
  const [parameters, setParameters] = useState(144e9) // 144B
  const [tokens, setTokens] = useState(12e12) // 12T
  
  // Hardware Parameters
  const [numNodes, setNumNodes] = useState(72)
  const [pflopsPerNode, setPflopsPerNode] = useState(32) // GH200 x16
  const [bandwidthMbps, setBandwidthMbps] = useState(100)
  const [mfu, setMfu] = useState(0.4)

  // Algorithm Parameters
  const [innerSteps, setInnerSteps] = useState(128)
  const [compression, setCompression] = useState(16)
  const [localBatch, setLocalBatch] = useState(131072)

  const [results, setResults] = useState<any>(null)

  useEffect(() => {
    calculate()
  }, [parameters, tokens, numNodes, pflopsPerNode, bandwidthMbps, mfu, innerSteps, compression, localBatch])

  const calculate = () => {
    // 1. Compute Time per Inner Step (per node)
    // FLOPs per inner step = 6 * P * Local_Tokens
    const flopsPerStep = 6 * parameters * localBatch
    const nodeComputePower = pflopsPerNode * 1e15 * mfu
    const timePerInnerStep = flopsPerStep / nodeComputePower
    
    // 2. Communication Time per Outer Step (per node)
    // payload = params * precision(2 bytes) * 8 bits / compression
    const payloadBits = (parameters * 2 * 8) / compression
    const totalCommBits = 2 * payloadBits // Up + Down
    const commTimePerOuterStep = totalCommBits / (bandwidthMbps * 1e6)
    
    // 3. Total Time (Assuming Streaming/Overlap)
    const computeBlockTime = innerSteps * timePerInnerStep
    const effectiveOuterTime = Math.max(computeBlockTime, commTimePerOuterStep)

    // Total Steps
    const globalBatchTokens = localBatch * numNodes
    const totalGlobalSteps = tokens / globalBatchTokens
    const totalOuterSteps = totalGlobalSteps / innerSteps
    
    const totalTimeSeconds = totalOuterSteps * effectiveOuterTime
    const days = totalTimeSeconds / (24 * 3600)

    setResults({
      computeStepSec: timePerInnerStep.toFixed(2),
      commOuterSec: commTimePerOuterStep.toFixed(2),
      computeBlockSec: computeBlockTime.toFixed(2),
      days: days.toFixed(2),
      bottleneck: commTimePerOuterStep > computeBlockTime ? "Network" : "Compute",
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
            <input type="range" min="1" max="2000" step="1" value={parameters / 1e9} onChange={(e) => setParameters(Number(e.target.value) * 1e9)} />
          </div>
          <div className="input-group">
            <label>Tokens (T): {tokens / 1e12}</label>
            <input type="range" min="0.1" max="50" step="0.1" value={tokens / 1e12} onChange={(e) => setTokens(Number(e.target.value) * 1e12)} />
          </div>
        </section>

        <section>
          <h3>Infrastructure</h3>
          <div className="input-group">
            <label>Number of Nodes: {numNodes}</label>
            <input type="range" min="1" max="1000" step="1" value={numNodes} onChange={(e) => setNumNodes(Number(e.target.value))} />
          </div>
          <div className="input-group">
            <label>Node PFLOPS: {pflopsPerNode}</label>
            <input type="range" min="1" max="500" step="1" value={pflopsPerNode} onChange={(e) => setPflopsPerNode(Number(e.target.value))} />
          </div>
          <div className="input-group">
            <label>WAN Bandwidth (Mbps): {bandwidthMbps}</label>
            <input type="range" min="1" max="10000" step="10" value={bandwidthMbps} onChange={(e) => setBandwidthMbps(Number(e.target.value))} />
          </div>
        </section>

        <section>
          <h3>Algorithm (DiLoCo)</h3>
          <div className="input-group">
            <label>Inner Steps: {innerSteps}</label>
            <input type="range" min="1" max="1000" step="1" value={innerSteps} onChange={(e) => setInnerSteps(Number(e.target.value))} />
          </div>
          <div className="input-group">
            <label>Compression Ratio: {compression}x</label>
            <input type="range" min="1" max="100" step="1" value={compression} onChange={(e) => setCompression(Number(e.target.value))} />
          </div>
        </section>

        {results && (
          <div className="results-card">
            <h2>Results</h2>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
              <div>
                <p><strong>Total Training Time:</strong></p>
                <h1 style={{ color: results.days < 365 ? '#4caf50' : '#ff4444', margin: 0 }}>{results.days} Days</h1>
                <p>{results.feasibility}</p>
              </div>
              <div>
                <p><strong>Bottleneck:</strong> {results.bottleneck}</p>
                <p>Compute Block: {results.computeBlockSec}s</p>
                <p>Network Sync: {results.commOuterSec}s</p>
              </div>
            </div>
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
