import { useState, useEffect } from 'react'

const GrowthInput = ({ label, value, onChange }: { label: string, value: number, onChange: (val: number) => void }) => {
  const multiple = Math.pow(10, value);
  const percent = (multiple - 1) * 100;

  const inputStyle = {
    background: '#1a1a1a',
    color: 'white',
    border: '1px solid #646cff',
    padding: '5px',
    borderRadius: '4px',
    width: '100%',
    boxSizing: 'border-box' as const
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '140px 1fr 1fr 1fr', gap: '10px', alignItems: 'end', marginBottom: '15px' }}>
      <label style={{ paddingBottom: '8px', fontSize: '0.9em' }}>{label}</label>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        <span style={{ fontSize: '0.7em', color: '#aaa' }}>OOM/yr</span>
        <input 
          type="number" 
          step="0.01" 
          value={parseFloat(value.toFixed(3))} 
          onChange={(e) => onChange(Number(e.target.value))} 
          style={inputStyle} 
        />
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        <span style={{ fontSize: '0.7em', color: '#aaa' }}>%/yr</span>
        <input 
          type="number" 
          step="1" 
          value={Math.round(percent)} 
          onChange={(e) => onChange(Math.log10(Number(e.target.value) / 100 + 1))} 
          style={inputStyle} 
        />
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        <span style={{ fontSize: '0.7em', color: '#aaa' }}>Multiple</span>
        <input 
          type="number" 
          step="0.1" 
          value={parseFloat(multiple.toFixed(2))} 
          onChange={(e) => onChange(Math.log10(Number(e.target.value)))} 
          style={inputStyle} 
        />
      </div>
    </div>
  );
};

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
  const [precision, setPrecision] = useState('FP16')

  // The Longest Training Run Calculator
  const [showLongestRunCalc, setShowLongestRunCalc] = useState(false)
  const [hwGrowth, setHwGrowth] = useState(0.37)
  const [swGrowth, setSwGrowth] = useState(2.0)
  const [investGrowth, setInvestGrowth] = useState(2.5)

  const [results, setResults] = useState<any>(null)

  useEffect(() => {
    // Sync compression based on precision
    if (precision === 'FP8') {
      setCompression(2)
      setPpCompression(2)
    } else if (precision === 'FP4') {
      setCompression(4)
      setPpCompression(4)
    } else {
      setCompression(1)
      setPpCompression(1)
    }
  }, [precision])

  useEffect(() => {
    calculate()
  }, [parameters, tokens, numNodes, pflopsPerNode, vramPerNode, bandwidthMbps, latencyMs, mfu, innerSteps, compression, localBatch, ppCompression, microBatches, useHierarchy, nodesPerGroup, regionalBandwidth, regionalLatency, regionalSteps, hwGrowth, swGrowth, investGrowth])

  const calculate = () => {
    // 1. Memory Analysis
    const bytesPerParam = precision === 'FP16' ? 12 : precision === 'FP8' ? 8 : 6
    const isSharded = (parameters * bytesPerParam) > (vramPerNode * 1e9)
    const ppStages = Math.ceil((parameters * bytesPerParam) / (vramPerNode * 1e9))
    
    // 2. Compute Time
    const flopsPerStep = 6 * parameters * localBatch
    const nodeComputePower = pflopsPerNode * 1e15 * mfu
    const computeTimePerStep = flopsPerStep / nodeComputePower
    
    // 3. Algorithmic Efficiency Penalty
    // Research (Wang et al. 2018) suggests hierarchy "anchors" drift.
    // The effective H for penalty is less than total steps because regional syncs reset drift partially.
    const effectiveH = useHierarchy 
      ? innerSteps * Math.pow(regionalSteps, 0.5) // Hierarchical benefit (drift anchoring)
      : innerSteps
    
    // Alpha reduces slightly for larger models (more robust)
    const alpha = 0.08 * (1 / (1 + Math.log10(parameters / 1e9) / 5))
    const algorithmicEfficiency = Math.max(0.5, 1 - alpha * Math.log10(effectiveH))
    
    // 4. Straggler & Congestion Penalty
    // Scaling to more nodes increases the chance of a P99 latency spike (straggler)
    const getStragglerFactor = (n: number) => 1 + 0.05 * Math.log2(n)
    
    let totalTimeSeconds = 0
    let mode = "Data Parallel (DiLoCo)"
    let globalCommSec = 0
    let regionalCommSec = 0
    let computeBlockSec = 0
    let latencyPenaltySec = 0

    if (!isSharded) {
      // --- Data Parallel Mode ---
      const payloadBits = (parameters * 2 * 8) / compression
      
      if (useHierarchy) {
        mode = "Hierarchical DiLoCo"
        const numGroups = numNodes / nodesPerGroup
        
        // Regional Sync (Every 'innerSteps')
        regionalCommSec = ((2 * payloadBits) / (regionalBandwidth * 1e6) + (regionalLatency / 1000)) * getStragglerFactor(nodesPerGroup)
        
        // Global Sync (Every 'totalInnerSteps') - Only leaders communicate
        globalCommSec = ((2 * payloadBits) / (bandwidthMbps * 1e6) + (latencyMs / 1000)) * getStragglerFactor(numGroups)
        
        // One global cycle = H_global regional cycles
        const regionalCycleTime = Math.max(innerSteps * computeTimePerStep, regionalCommSec)
        const globalCycleTime = Math.max(regionalSteps * regionalCycleTime, globalCommSec)
        
        const totalGlobalCycles = (tokens / (localBatch * numNodes)) / (innerSteps * regionalSteps)
        totalTimeSeconds = totalGlobalCycles * globalCycleTime
        computeBlockSec = (innerSteps * regionalSteps) * computeTimePerStep
      } else {
        computeBlockSec = innerSteps * computeTimePerStep
        globalCommSec = ((2 * payloadBits) / (bandwidthMbps * 1e6) + (latencyMs / 1000)) * getStragglerFactor(numNodes)
        
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

      // Straggler penalty applies to the pipeline handoff as well
      const straggler = getStragglerFactor(numNodes / ppStages)
      const timePerStep = (microBatches + ppStages - 1) * (computePerMicroSec + (commPerMicroSec + latencySec) * straggler)
      
      globalCommSec = (microBatches + ppStages - 1) * commPerMicroSec * straggler
      latencyPenaltySec = (microBatches + ppStages - 1) * latencySec * straggler
      computeBlockSec = (microBatches + ppStages - 1) * computePerMicroSec
      
      totalTimeSeconds = (tokens / (localBatch * (numNodes / ppStages))) * timePerStep
    }

    // Effective compute time accounts for algorithmic penalty
    const totalTimeDays = totalTimeSeconds / (24 * 3600)
    const effectiveDays = totalTimeDays / algorithmicEfficiency
    const effectiveSeconds = effectiveDays * 24 * 3600

    // Global Utilization Metrics (End-to-End)
    // Theoretical FLOPs = 6 * parameters * tokens
    // Hardware Max FLOPs = numNodes * pflopsPerNode * 1e15 * effectiveSeconds
    const theoreticalFlops = 6 * parameters * (tokens)
    const hardwareMaxFlops = numNodes * (pflopsPerNode * 1e15) * effectiveSeconds
    const globalMfu = (theoreticalFlops / hardwareMaxFlops)
    const globalHfu = globalMfu / 0.8 // MFU is ~80% of HFU per research

    // Calculate Dynamic Max Training Run (Epoch - The Longest Training Run)
    // Formula: L = 1 / (gH + gS + gI)
    const combinedGrowth = hwGrowth + swGrowth + investGrowth
    const maxDays = (1 / combinedGrowth) * 365

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
      maxDays: maxDays.toFixed(0),
      bottleneck: (globalCommSec + latencyPenaltySec) > computeBlockSec ? "Network" : "Compute",
      feasibility: effectiveDays < maxDays ? "Feasible" : `Impractical (>${maxDays.toFixed(0)} days)`
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
            <input 
              type="number" 
              min="1" 
              max="15000" 
              step="10" 
              value={parameters / 1e9} 
              onChange={(e) => setParameters(Number(e.target.value) * 1e9)} 
              style={{ background: '#1a1a1a', color: 'white', border: '1px solid #646cff', padding: '5px', borderRadius: '4px' }}
            />
            <span>B</span>
          </div>
          <div className="input-group">
            <label>Tokens (T):</label>
            <input 
              type="number" 
              min="0.1" 
              max="100" 
              step="0.1" 
              value={tokens / 1e12} 
              onChange={(e) => setTokens(Number(e.target.value) * 1e12)} 
              style={{ background: '#1a1a1a', color: 'white', border: '1px solid #646cff', padding: '5px', borderRadius: '4px' }}
            />
            <span>T</span>
          </div>
        </section>

        <section>
          <h3>Infrastructure (Global)</h3>
          <div className="input-group">
            <label>Number of Nodes:</label>
            <input 
              type="number" 
              min="1" 
              max="5000" 
              step="1" 
              value={numNodes} 
              onChange={(e) => setNumNodes(Number(e.target.value))} 
              style={{ background: '#1a1a1a', color: 'white', border: '1px solid #646cff', padding: '5px', borderRadius: '4px' }}
            />
            <span>Nodes</span>
          </div>
          <div className="input-group">
            <label>WAN Bandwidth (Mbps):</label>
            <input 
              type="number" 
              min="1" 
              max="10000" 
              step="10" 
              value={bandwidthMbps} 
              onChange={(e) => setBandwidthMbps(Number(e.target.value))} 
              style={{ background: '#1a1a1a', color: 'white', border: '1px solid #646cff', padding: '5px', borderRadius: '4px' }}
            />
            <span>Mbps</span>
          </div>
          <div className="input-group">
            <label>WAN Latency (ms):</label>
            <input 
              type="number" 
              min="1" 
              max="1000" 
              step="10" 
              value={latencyMs} 
              onChange={(e) => setLatencyMs(Number(e.target.value))} 
              style={{ background: '#1a1a1a', color: 'white', border: '1px solid #646cff', padding: '5px', borderRadius: '4px' }}
            />
            <span>ms</span>
          </div>
          <div className="input-group">
            <label>Node VRAM (GB):</label>
            <input 
              type="number" 
              min="8" 
              max="5000" 
              step="8" 
              value={vramPerNode} 
              onChange={(e) => setVramPerNode(Number(e.target.value))} 
              style={{ background: '#1a1a1a', color: 'white', border: '1px solid #646cff', padding: '5px', borderRadius: '4px' }}
            />
            <span>GB</span>
          </div>
          <div className="input-group">
            <label>Node PFLOPS:</label>
            <input 
              type="number" 
              min="1" 
              max="1000" 
              step="1" 
              value={pflopsPerNode} 
              onChange={(e) => setPflopsPerNode(Number(e.target.value))} 
              style={{ background: '#1a1a1a', color: 'white', border: '1px solid #646cff', padding: '5px', borderRadius: '4px' }}
            />
            <span>PFLOPS</span>
          </div>
          <div className="input-group">
            <label>Base MFU (%):</label>
            <input 
              type="number" 
              min="5" 
              max="80" 
              step="1" 
              value={mfu * 100} 
              onChange={(e) => setMfu(Number(e.target.value) / 100)} 
              style={{ background: '#1a1a1a', color: 'white', border: '1px solid #646cff', padding: '5px', borderRadius: '4px' }}
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
            <h3>Hierarchical Sync</h3>
            <input type="checkbox" checked={useHierarchy} onChange={(e) => setUseHierarchy(e.target.checked)} />
          </div>
          {useHierarchy && (
            <div style={{ borderLeft: '2px solid #646cff', paddingLeft: '15px' }}>
              <div className="input-group">
                <label>Group Nodes:</label>
                <input 
                  type="number" 
                  min="2" 
                  max={numNodes} 
                  step="1" 
                  value={nodesPerGroup} 
                  onChange={(e) => setNodesPerGroup(Number(e.target.value))} 
                  style={{ background: '#1a1a1a', color: 'white', border: '1px solid #646cff', padding: '5px', borderRadius: '4px' }}
                />
                <span>Nodes</span>
              </div>
              <div className="input-group">
                <label>Regional Bandwidth (Mbps):</label>
                <input 
                  type="number" 
                  min="100" 
                  max="100000" 
                  step="100" 
                  value={regionalBandwidth} 
                  onChange={(e) => setRegionalBandwidth(Number(e.target.value))} 
                  style={{ background: '#1a1a1a', color: 'white', border: '1px solid #646cff', padding: '5px', borderRadius: '4px' }}
                />
                <span>Mbps</span>
              </div>
              <div className="input-group">
                <label>Regional Latency (ms):</label>
                <input 
                  type="number" 
                  min="1" 
                  max="100" 
                  step="1" 
                  value={regionalLatency} 
                  onChange={(e) => setRegionalLatency(Number(e.target.value))} 
                  style={{ background: '#1a1a1a', color: 'white', border: '1px solid #646cff', padding: '5px', borderRadius: '4px' }}
                />
                <span>ms</span>
              </div>
              <div className="input-group">
                <label>Regional Sync Steps:</label>
                <input 
                  type="number" 
                  min="1" 
                  max="100" 
                  step="1" 
                  value={regionalSteps} 
                  onChange={(e) => setRegionalSteps(Number(e.target.value))} 
                  style={{ background: '#1a1a1a', color: 'white', border: '1px solid #646cff', padding: '5px', borderRadius: '4px' }}
                />
                <span>Steps</span>
              </div>
            </div>
          )}
        </section>

        <section>
          <h3>Algorithm Settings</h3>
          <div className="input-group">
            <label>Precision:</label>
            <select value={precision} onChange={(e) => setPrecision(e.target.value)} style={{ background: '#1a1a1a', color: 'white', border: '1px solid #646cff', padding: '5px', borderRadius: '4px' }}>
              <option value="FP16">FP16 / BF16 (2 bytes)</option>
              <option value="FP8">FP8 (1 byte)</option>
              <option value="FP4">FP4 (0.5 byte)</option>
            </select>
            <span>{precision}</span>
          </div>
          <div className="input-group">
            <label>Inner Steps (Local):</label>
            <input 
              type="number" 
              min="1" 
              max="1000" 
              step="1" 
              value={innerSteps} 
              onChange={(e) => setInnerSteps(Number(e.target.value))} 
              style={{ background: '#1a1a1a', color: 'white', border: '1px solid #646cff', padding: '5px', borderRadius: '4px' }}
            />
            <span>Steps</span>
          </div>
          <div className="input-group">
            <label>Weight Compression:</label>
            <input 
              type="number" 
              min="1" 
              max="100" 
              step="1" 
              value={compression} 
              onChange={(e) => setCompression(Number(e.target.value))} 
              style={{ background: '#1a1a1a', color: 'white', border: '1px solid #646cff', padding: '5px', borderRadius: '4px' }}
            />
            <span>x</span>
          </div>
          <div className="input-group">
            <label>Activation Compression:</label>
            <input 
              type="number" 
              min="1" 
              max="100" 
              step="1" 
              value={ppCompression} 
              onChange={(e) => setPpCompression(Number(e.target.value))} 
              style={{ background: '#1a1a1a', color: 'white', border: '1px solid #646cff', padding: '5px', borderRadius: '4px' }}
            />
            <span>x</span>
          </div>
        </section>

        <section>
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center', cursor: 'pointer' }} onClick={() => setShowLongestRunCalc(!showLongestRunCalc)}>
            <h3>Max Run Duration Calculator {showLongestRunCalc ? '▼' : '▶'}</h3>
          </div>
          <p style={{ fontSize: '0.8em', color: '#aaa', marginTop: '-10px' }}>
            Based on Epoch's "The Longest Training Run" research.
          </p>
          {showLongestRunCalc && (
            <div style={{ borderLeft: '2px solid #646cff', paddingLeft: '15px', marginBottom: '20px', marginTop: '10px' }}>
              <GrowthInput label="HW Growth" value={hwGrowth} onChange={setHwGrowth} />
              <GrowthInput label="SW Growth" value={swGrowth} onChange={setSwGrowth} />
              <GrowthInput label="Invest Growth" value={investGrowth} onChange={setInvestGrowth} />
              
              <p style={{ fontSize: '0.75em', color: '#888', marginTop: '10px' }}>
                * High growth rates suggest shorter optimal runs, as delaying for better tech/budget becomes more attractive.
                Calculated as L = 1 / Σ(OOM/yr).
              </p>
            </div>
          )}
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
                <p style={{ fontSize: '0.8em', color: '#aaa' }}>
                  Global MFU: <span style={{ color: results.globalMfu < 20 ? '#ff4444' : '#aaa' }}>{results.globalMfu}%</span> | 
                  HFU: {results.globalHfu}%
                </p>
                {results.globalMfu < 20 && (
                  <p style={{ fontSize: '0.7em', color: '#ff4444' }}>
                    * Inefficient run: Communication overhead or algorithmic penalty is dominating.
                  </p>
                )}
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
