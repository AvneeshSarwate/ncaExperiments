import Foundation
import Metal

// MARK: - Constants
let C = 16, HIDDEN = 128, GS = 72, TS = 40, TPAD = 16, PERC = C * 3
let STATE = C * GS * GS
let HW = GS * GS

// MARK: - Metal Setup
class NCAMetal {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary

    // Forward pipelines
    let perceivePSO: MTLComputePipelineState
    let biasReluPSO: MTLComputePipelineState
    let maxPool3x3PSO: MTLComputePipelineState
    let thresholdMaskPSO: MTLComputePipelineState
    let maskedResidualPSO: MTLComputePipelineState
    let sliceAlphaPSO: MTLComputePipelineState

    // 1x1 conv pipelines
    let conv1x1FwdPSO: MTLComputePipelineState
    let conv1x1DataGradPSO: MTLComputePipelineState
    let conv1x1WeightGradPSO: MTLComputePipelineState

    // Backward pipelines
    let bwdMaskedResidualPSO: MTLComputePipelineState
    let reluBackwardPSO: MTLComputePipelineState
    let biasGradPSO: MTLComputePipelineState
    let perceiveBackwardPSO: MTLComputePipelineState
    let elemAddPSO: MTLComputePipelineState

    init() {
        device = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first!
        queue = device.makeCommandQueue()!
        // Find metallib relative to executable or cwd
        print("Loading Metal library...")
        let cwd = FileManager.default.currentDirectoryPath
        let libPath = cwd + "/nca_kernels.metallib"
        print("Path: \(libPath)")
        do {
            library = try device.makeLibrary(URL: URL(fileURLWithPath: libPath))
        } catch {
            fatalError("Failed to load \(libPath): \(error)")
        }

        let lib = library
        let dev = device
        func pso(_ name: String) -> MTLComputePipelineState {
            guard let fn = lib.makeFunction(name: name) else { fatalError("No function: \(name)") }
            do { return try dev.makeComputePipelineState(function: fn) }
            catch { fatalError("PSO failed for \(name): \(error)") }
        }
        conv1x1FwdPSO = pso("conv1x1_forward")
        conv1x1DataGradPSO = pso("conv1x1_data_grad")
        conv1x1WeightGradPSO = pso("conv1x1_weight_grad")
        perceivePSO = pso("perceive")
        biasReluPSO = pso("bias_relu")
        maxPool3x3PSO = pso("max_pool_3x3")
        thresholdMaskPSO = pso("threshold_mask")
        maskedResidualPSO = pso("masked_residual")
        sliceAlphaPSO = pso("slice_alpha")
        bwdMaskedResidualPSO = pso("backward_masked_residual")
        reluBackwardPSO = pso("relu_backward")
        biasGradPSO = pso("bias_grad")
        perceiveBackwardPSO = pso("perceive_backward")
        elemAddPSO = pso("elementwise_add")
    }

    func makeBuffer(_ data: [Float]) -> MTLBuffer {
        device.makeBuffer(bytes: data, length: data.count * 4, options: .storageModeShared)!
    }

    func makeBuffer(size: Int) -> MTLBuffer {
        device.makeBuffer(length: size * 4, options: .storageModeShared)!
    }

    func readBuffer(_ buf: MTLBuffer, count: Int) -> [Float] {
        let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    // ---- 1x1 conv dispatch (NCHW, no reshape) ----
    func dispatchConv1x1Fwd(_ enc: MTLComputeCommandEncoder, input: MTLBuffer, weight: MTLBuffer,
                             output: MTLBuffer, Cin: Int, Cout: Int) {
        enc.setComputePipelineState(conv1x1FwdPSO)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(weight, offset: 0, index: 1)
        enc.setBuffer(output, offset: 0, index: 2)
        var cin = Int32(Cin); enc.setBytes(&cin, length: 4, index: 3)
        var cout = Int32(Cout); enc.setBytes(&cout, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: GS, height: GS, depth: 8 * Cout),
                            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
    }

    // ---- Forward pass (one NCA step) ----
    // Returns: (output, perc, hidden, fireMask, lifeMask) buffers
    // fireMask is all-ones when fireRate=1.0
    func forwardStep(_ cmd: MTLCommandBuffer, state: MTLBuffer,
                     fc1W: MTLBuffer, fc1B: MTLBuffer, fc2W: MTLBuffer,
                     fireRate: Float = 1.0) -> (MTLBuffer, MTLBuffer, MTLBuffer, MTLBuffer, MTLBuffer) {
        let B = 8
        let percBuf = makeBuffer(size: B * PERC * HW)
        let fc1OutBuf = makeBuffer(size: B * HIDDEN * HW)
        let hiddenBuf = makeBuffer(size: B * HIDDEN * HW)
        let deltaBuf = makeBuffer(size: B * C * HW)
        let alphaPre = makeBuffer(size: B * HW)
        let maxAlphaPre = makeBuffer(size: B * HW)
        let preMask = makeBuffer(size: B * HW)
        let alphaPost = makeBuffer(size: B * HW)
        let maxAlphaPost = makeBuffer(size: B * HW)
        let postMask = makeBuffer(size: B * HW)
        let lifeMask = makeBuffer(size: B * HW)
        let fireMask = makeBuffer(size: B * HW)
        let output = makeBuffer(size: B * C * HW)

        // Fill fireMask (all 1s for fireRate=1.0, random otherwise)
        if fireRate >= 1.0 {
            let ptr = fireMask.contents().bindMemory(to: Float.self, capacity: B * HW)
            for i in 0..<(B * HW) { ptr[i] = 1.0 }
        } else {
            let ptr = fireMask.contents().bindMemory(to: Float.self, capacity: B * HW)
            for i in 0..<(B * HW) { ptr[i] = Float.random(in: 0...1) < fireRate ? 1.0 : 0.0 }
        }

        // 1. Perceive
        var enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(perceivePSO)
        enc.setBuffer(state, offset: 0, index: 0)
        enc.setBuffer(percBuf, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: GS, height: GS, depth: B * PERC),
                            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        enc.endEncoding()

        // 2. FC1: 1x1 conv (PERC→HIDDEN)
        enc = cmd.makeComputeCommandEncoder()!
        dispatchConv1x1Fwd(enc, input: percBuf, weight: fc1W, output: fc1OutBuf, Cin: PERC, Cout: HIDDEN)
        enc.endEncoding()

        // 3. Bias + ReLU
        enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(biasReluPSO)
        enc.setBuffer(fc1OutBuf, offset: 0, index: 0)
        enc.setBuffer(fc1B, offset: 0, index: 1)
        enc.setBuffer(hiddenBuf, offset: 0, index: 2)
        enc.dispatchThreads(MTLSize(width: GS, height: GS, depth: B * HIDDEN),
                            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        enc.endEncoding()

        // 4. FC2: 1x1 conv (HIDDEN→C)
        enc = cmd.makeComputeCommandEncoder()!
        dispatchConv1x1Fwd(enc, input: hiddenBuf, weight: fc2W, output: deltaBuf, Cin: HIDDEN, Cout: C)
        enc.endEncoding()

        // 5. Pre-alive mask
        enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(sliceAlphaPSO)
        enc.setBuffer(state, offset: 0, index: 0)
        enc.setBuffer(alphaPre, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: GS, height: GS, depth: B),
                            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        enc.endEncoding()

        enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(maxPool3x3PSO)
        enc.setBuffer(alphaPre, offset: 0, index: 0)
        enc.setBuffer(maxAlphaPre, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: GS, height: GS, depth: B),
                            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        enc.endEncoding()

        enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(thresholdMaskPSO)
        enc.setBuffer(maxAlphaPre, offset: 0, index: 0)
        enc.setBuffer(preMask, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: B * HW, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()

        // 6. Compute updated = state + delta, then post-alive mask, then apply both masks
        // First: slice alpha from updated (need updated first)
        // We'll use masked_residual which computes (state + delta * fireMask) * lifeMask
        // But we need lifeMask = preMask * postMask, and postMask depends on updated.
        // So: compute updated first (with fireMask, without lifeMask), then post-mask, then apply.

        // Compute updated = state + delta * fireMask (store temporarily in output buffer)
        // We can use masked_residual with lifeMask=all-ones for this step
        let onesMask = makeBuffer(size: B * HW)
        let onesPtr = onesMask.contents().bindMemory(to: Float.self, capacity: B * HW)
        for i in 0..<(B * HW) { onesPtr[i] = 1.0 }

        enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(maskedResidualPSO)
        enc.setBuffer(state, offset: 0, index: 0)
        enc.setBuffer(deltaBuf, offset: 0, index: 1)
        enc.setBuffer(fireMask, offset: 0, index: 2)
        enc.setBuffer(onesMask, offset: 0, index: 3)  // lifeMask = 1.0
        enc.setBuffer(output, offset: 0, index: 4)    // temp: updated = state + delta * fireMask
        enc.dispatchThreads(MTLSize(width: GS, height: GS, depth: B * C),
                            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        enc.endEncoding()

        // Post-alive mask on updated
        enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(sliceAlphaPSO)
        enc.setBuffer(output, offset: 0, index: 0)  // updated
        enc.setBuffer(alphaPost, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: GS, height: GS, depth: B),
                            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        enc.endEncoding()

        enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(maxPool3x3PSO)
        enc.setBuffer(alphaPost, offset: 0, index: 0)
        enc.setBuffer(maxAlphaPost, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: GS, height: GS, depth: B),
                            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        enc.endEncoding()

        enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(thresholdMaskPSO)
        enc.setBuffer(maxAlphaPost, offset: 0, index: 0)
        enc.setBuffer(postMask, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: B * HW, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()

        // lifeMask = preMask * postMask (element-wise, reuse as multiply)
        // Then final output = updated * lifeMask
        // Redo masked_residual with proper lifeMask
        // lifeMask = preMask * postMask
        let lifeMaskFinal = makeBuffer(size: B * HW)
        enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(thresholdMaskPSO) // reuse: just multiply two masks
        // Actually need a multiply kernel. Use elementwise approach:
        enc.endEncoding()

        // Multiply preMask * postMask → lifeMask
        // I don't have a dedicated mask-multiply kernel. Let me just compute inline:
        // Actually I can recompute: output = (state + delta * fireMask) * (preMask * postMask)
        // Use masked_residual again with the product mask
        // But masked_residual takes a single lifeMask. Let me compute preMask * postMask first.
        // I'll abuse threshold_mask... no. Let me just compute it on CPU for now and fix later.
        // Actually, elementwise_add won't work for multiply. Let me add the result differently.

        // SIMPLEST: read preMask and postMask, multiply on CPU, upload.
        // This is a 41KB tensor, negligible.
        cmd.commit()
        cmd.waitUntilCompleted()

        let preData = readBuffer(preMask, count: B * HW)
        let postData = readBuffer(postMask, count: B * HW)
        let updatedData = readBuffer(output, count: B * C * HW)
        var lifeData = [Float](repeating: 0, count: B * HW)
        for i in 0..<(B * HW) { lifeData[i] = preData[i] * postData[i] }

        let lifeBuf = makeBuffer(lifeData)

        // Apply lifeMask to updated → final output
        var finalData = updatedData
        for b in 0..<B {
            for c in 0..<C {
                for yx in 0..<HW {
                    finalData[(b * C + c) * HW + yx] *= lifeData[b * HW + yx]
                }
            }
        }

        let finalBuf = makeBuffer(finalData)

        return (finalBuf, percBuf, hiddenBuf, fireMask, lifeBuf)
    }
}

// MARK: - Verification
func loadRef(_ name: String) -> [Float] {
    let data = try! Data(contentsOf: URL(fileURLWithPath: "reference_ops/\(name).bin"))
    return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

func compare(_ name: String, _ gpu: [Float], _ ref: [Float]) {
    guard gpu.count == ref.count else { print("  \(name): SIZE MISMATCH \(gpu.count) vs \(ref.count)"); return }
    var maxErr: Float = 0
    var maxIdx = 0
    for i in 0..<gpu.count {
        let e = abs(gpu[i] - ref[i])
        if e > maxErr { maxErr = e; maxIdx = i }
    }
    let status = maxErr < 1e-4 ? "PASS" : maxErr < 1e-2 ? "MARGINAL" : "FAIL"
    print(String(format: "  %-15s max_err=%.3e idx=%d  %@", name, maxErr, maxIdx, status))
}

// MARK: - Main
import Darwin
fputs("Starting...\n", stderr)
let metal = NCAMetal()
fputs("Metal initialized\n", stderr)
print("Device: \(metal.device.name)")

fputs("Loading weights...\n", stderr)
// Load weights
let weightsData = try! Data(contentsOf: URL(fileURLWithPath: "../output/A/weights.bin"))
let allW: [Float] = weightsData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
let fc1WData = Array(allW[0..<HIDDEN*PERC])
let fc1BData = Array(allW[HIDDEN*PERC..<HIDDEN*PERC+HIDDEN])
let fc2WData = Array(allW[HIDDEN*PERC+HIDDEN..<HIDDEN*PERC+HIDDEN+C*HIDDEN])
fputs("Creating buffers...\n", stderr)

let fc1W = metal.makeBuffer(fc1WData)
let fc1B = metal.makeBuffer(fc1BData)
let fc2W = metal.makeBuffer(fc2WData)

fputs("Loading refs...\n", stderr)
// Load input reference
let inputData = loadRef("input")
let stateBuf = metal.makeBuffer(inputData)

// Run forward step
print("\n=== Forward Pass Verification ===")
let cmd = metal.queue.makeCommandBuffer()!
let (outBuf, percBuf, hiddenBuf, fmBuf, lmBuf) = metal.forwardStep(cmd, state: stateBuf,
    fc1W: fc1W, fc1B: fc1B, fc2W: fc2W, fireRate: 1.0)

// Note: forwardStep already committed. Read results directly.
let percOut = metal.readBuffer(percBuf, count: 8 * PERC * HW)
let hiddenOut = metal.readBuffer(hiddenBuf, count: 8 * HIDDEN * HW)
let outputOut = metal.readBuffer(outBuf, count: 8 * C * HW)

compare("perc", percOut, loadRef("perc"))
compare("hidden", hiddenOut, loadRef("hidden"))
compare("output", outputOut, loadRef("output"))

print("\nDone.")
