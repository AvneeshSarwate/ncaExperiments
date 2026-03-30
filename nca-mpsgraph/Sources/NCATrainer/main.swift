import Foundation
import Metal
import Darwin

// MARK: - Constants
let BATCH = 8
let C = 16, HIDDEN = 128, GS = 72, TS = 40, TPAD = 16, PERC = C * 3
let STATE = C * GS * GS
let HW = GS * GS

struct ForwardStepBuffers {
    let perc: MTLBuffer
    let fc1Out: MTLBuffer
    let hidden: MTLBuffer
    let delta: MTLBuffer
    let maxAlphaPre: MTLBuffer
    let preMask: MTLBuffer
    let updated: MTLBuffer
    let maxAlphaPost: MTLBuffer
    let postMask: MTLBuffer
    let lifeMask: MTLBuffer
    let fireMask: MTLBuffer
    let output: MTLBuffer
}

struct BackwardStepBuffers {
    let dUpdated: MTLBuffer
    let dDelta: MTLBuffer
    let dHidden: MTLBuffer
    let dFC2W: MTLBuffer
    let dFC1Raw: MTLBuffer
    let dFC1B: MTLBuffer
    let dPerc: MTLBuffer
    let dFC1W: MTLBuffer
    let dStateFromPerc: MTLBuffer
    let dState: MTLBuffer
}

struct RolloutScratch {
    let perc: MTLBuffer
    let hidden: MTLBuffer
    let delta: MTLBuffer
    let preMask: MTLBuffer
    let postMask: MTLBuffer
}

struct FastTrainScratch {
    let perc: MTLBuffer
    let percHWMajor: MTLBuffer
    let hidden: MTLBuffer
    let hiddenHWMajor: MTLBuffer
    let delta: MTLBuffer
    let preMask: MTLBuffer
    let postMask: MTLBuffer
    let lifeMask: MTLBuffer
    let output: MTLBuffer
    let dResidual: MTLBuffer
    let dFC2W: MTLBuffer
    let dFC1Raw: MTLBuffer
    let dFC1B: MTLBuffer
    let dPerc: MTLBuffer
    let dFC1W: MTLBuffer
    let dStateFromPerc: MTLBuffer
    let dState: MTLBuffer
}

func log(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

// MARK: - Metal Setup
class NCAMetal {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary

    // Forward pipelines
    let perceivePSO: MTLComputePipelineState
    let perceiveX3PSO: MTLComputePipelineState
    let biasReluPSO: MTLComputePipelineState
    let maxPool3x3PSO: MTLComputePipelineState
    let thresholdMaskPSO: MTLComputePipelineState
    let maskMultiplyPSO: MTLComputePipelineState
    let maskedResidualPSO: MTLComputePipelineState
    let residualAddPSO: MTLComputePipelineState
    let sliceAlphaPSO: MTLComputePipelineState
    let applyLifeMaskPSO: MTLComputePipelineState
    let aliveMaskFromStatePSO: MTLComputePipelineState
    let applyTwoMasksPSO: MTLComputePipelineState
    let applyTwoMasksStoreLifePSO: MTLComputePipelineState
    let rolloutTwoStepsFusedPSO: MTLComputePipelineState

    // 1x1 conv pipelines
    let conv1x1FwdPSO: MTLComputePipelineState
    let conv1x1FwdX4PSO: MTLComputePipelineState
    let conv1x1FwdX8PSO: MTLComputePipelineState
    let conv1x1DataGradPSO: MTLComputePipelineState
    let conv1x1DataGradT4PSO: MTLComputePipelineState
    let conv1x1DataGradReluT4PSO: MTLComputePipelineState
    let fc1DataGradTiledPSO: MTLComputePipelineState
    let fc2DataGradReluTiledPSO: MTLComputePipelineState
    let packChannelsHWMajorPSO: MTLComputePipelineState
    let fc1WeightGradTiledPSO: MTLComputePipelineState
    let fc2WeightGradTiledPSO: MTLComputePipelineState
    let conv1x1WeightGradPSO: MTLComputePipelineState

    // Backward pipelines
    let bwdMaskedResidualPSO: MTLComputePipelineState
    let reluBackwardPSO: MTLComputePipelineState
    let biasGradPSO: MTLComputePipelineState
    let perceiveBackwardPSO: MTLComputePipelineState
    let elemAddPSO: MTLComputePipelineState

    init() {
        guard let device = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first else {
            fatalError("No Metal device available")
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            fatalError("Failed to create Metal command queue")
        }
        self.queue = queue

        // Find metallib relative to executable or cwd
        log("Loading Metal library...")
        let cwd = FileManager.default.currentDirectoryPath
        let libPath = cwd + "/nca_kernels.metallib"
        log("Path: \(libPath)")
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
        conv1x1FwdX4PSO = pso("conv1x1_forward_x4")
        conv1x1FwdX8PSO = pso("conv1x1_forward_x8")
        conv1x1DataGradPSO = pso("conv1x1_data_grad")
        fc1DataGradTiledPSO = pso("fc1_data_grad_tiled")
        fc2DataGradReluTiledPSO = pso("fc2_data_grad_relu_tiled")
        packChannelsHWMajorPSO = pso("pack_channels_hw_major")
        fc1WeightGradTiledPSO = pso("fc1_weight_grad_tiled")
        fc2WeightGradTiledPSO = pso("fc2_weight_grad_tiled")
        conv1x1WeightGradPSO = pso("conv1x1_weight_grad")
        perceivePSO = pso("perceive")
        perceiveX3PSO = pso("perceive_x3")
        biasReluPSO = pso("bias_relu")
        maxPool3x3PSO = pso("max_pool_3x3")
        thresholdMaskPSO = pso("threshold_mask")
        maskMultiplyPSO = pso("mask_multiply")
        maskedResidualPSO = pso("masked_residual")
        residualAddPSO = pso("residual_add")
        sliceAlphaPSO = pso("slice_alpha")
        applyLifeMaskPSO = pso("apply_life_mask")
        aliveMaskFromStatePSO = pso("alive_mask_from_state")
        applyTwoMasksPSO = pso("apply_two_masks")
        applyTwoMasksStoreLifePSO = pso("apply_two_masks_store_life")
        rolloutTwoStepsFusedPSO = pso("rollout_two_steps_fused")
        bwdMaskedResidualPSO = pso("backward_masked_residual")
        reluBackwardPSO = pso("relu_backward")
        biasGradPSO = pso("bias_grad")
        perceiveBackwardPSO = pso("perceive_backward")
        elemAddPSO = pso("elementwise_add")
        conv1x1DataGradT4PSO = pso("conv1x1_data_grad_t4")
        conv1x1DataGradReluT4PSO = pso("conv1x1_data_grad_relu_t4")
    }

    func makeBuffer(_ data: [Float]) -> MTLBuffer {
        device.makeBuffer(bytes: data, length: data.count * 4, options: .storageModeShared)!
    }

    func makePrivateBuffer(_ data: [Float]) -> MTLBuffer {
        makePrivateCopy(of: makeBuffer(data))
    }

    func makeBuffer(size: Int) -> MTLBuffer {
        device.makeBuffer(length: size * 4, options: .storageModeShared)!
    }

    func makeBuffer(size: Int, options: MTLResourceOptions) -> MTLBuffer {
        device.makeBuffer(length: size * 4, options: options)!
    }

    func readBuffer(_ buf: MTLBuffer, count: Int) -> [Float] {
        let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    func writeBuffer(_ buf: MTLBuffer, data: [Float]) {
        let ptr = buf.contents().bindMemory(to: Float.self, capacity: data.count)
        data.withUnsafeBufferPointer { src in
            ptr.update(from: src.baseAddress!, count: data.count)
        }
    }

    func makePrivateCopy(of source: MTLBuffer) -> MTLBuffer {
        let copy = makeBuffer(size: source.length / 4, options: .storageModePrivate)
        let cmd = queue.makeCommandBuffer()!
        let blit = cmd.makeBlitCommandEncoder()!
        blit.copy(from: source, sourceOffset: 0, to: copy, destinationOffset: 0, size: source.length)
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        if let error = cmd.error {
            fatalError("Private copy failed: \(error)")
        }
        return copy
    }

    func dispatchGrid(_ enc: MTLComputeCommandEncoder, width: Int, height: Int, depth: Int) {
        enc.dispatchThreads(
            MTLSize(width: width, height: height, depth: depth),
            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1)
        )
    }

    func dispatch1D(_ enc: MTLComputeCommandEncoder, count: Int) {
        enc.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
    }

    func dispatchThreadgroups(_ enc: MTLComputeCommandEncoder, width: Int, height: Int, depth: Int,
                              threadsPerThreadgroup: MTLSize) {
        enc.dispatchThreadgroups(
            MTLSize(width: width, height: height, depth: depth),
            threadsPerThreadgroup: threadsPerThreadgroup
        )
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
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH * Cout)
    }

    func dispatchConv1x1FwdX4(_ enc: MTLComputeCommandEncoder, input: MTLBuffer, weightT: MTLBuffer,
                              output: MTLBuffer, Cin: Int, Cout: Int) {
        precondition(Cout % 4 == 0, "conv1x1_forward_x4 requires Cout divisible by 4")
        enc.setComputePipelineState(conv1x1FwdX4PSO)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(weightT, offset: 0, index: 1)
        enc.setBuffer(output, offset: 0, index: 2)
        var cin = Int32(Cin); enc.setBytes(&cin, length: 4, index: 3)
        var cout = Int32(Cout); enc.setBytes(&cout, length: 4, index: 4)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH * (Cout / 4))
    }

    func dispatchConv1x1FwdX8(_ enc: MTLComputeCommandEncoder, input: MTLBuffer, weightT: MTLBuffer,
                              output: MTLBuffer, Cin: Int, Cout: Int) {
        precondition(Cout % 8 == 0, "conv1x1_forward_x8 requires Cout divisible by 8")
        enc.setComputePipelineState(conv1x1FwdX8PSO)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(weightT, offset: 0, index: 1)
        enc.setBuffer(output, offset: 0, index: 2)
        var cin = Int32(Cin); enc.setBytes(&cin, length: 4, index: 3)
        var cout = Int32(Cout); enc.setBytes(&cout, length: 4, index: 4)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH * (Cout / 8))
    }

    func dispatchConv1x1DataGrad(_ enc: MTLComputeCommandEncoder, dOutput: MTLBuffer, weight: MTLBuffer,
                                 dInput: MTLBuffer, Cin: Int, Cout: Int) {
        enc.setComputePipelineState(conv1x1DataGradPSO)
        enc.setBuffer(dOutput, offset: 0, index: 0)
        enc.setBuffer(weight, offset: 0, index: 1)
        enc.setBuffer(dInput, offset: 0, index: 2)
        var cin = Int32(Cin); enc.setBytes(&cin, length: 4, index: 3)
        var cout = Int32(Cout); enc.setBytes(&cout, length: 4, index: 4)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH * Cin)
    }

    func dispatchConv1x1DataGradT4(_ enc: MTLComputeCommandEncoder, dOutput: MTLBuffer, weightT: MTLBuffer,
                                   dInput: MTLBuffer, Cin: Int, Cout: Int) {
        precondition(Cout % 4 == 0, "conv1x1_data_grad_t4 requires Cout divisible by 4")
        enc.setComputePipelineState(conv1x1DataGradT4PSO)
        enc.setBuffer(dOutput, offset: 0, index: 0)
        enc.setBuffer(weightT, offset: 0, index: 1)
        enc.setBuffer(dInput, offset: 0, index: 2)
        var cin = Int32(Cin); enc.setBytes(&cin, length: 4, index: 3)
        var cout = Int32(Cout); enc.setBytes(&cout, length: 4, index: 4)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH * Cin)
    }

    func dispatchConv1x1DataGradReluT4(_ enc: MTLComputeCommandEncoder, dOutput: MTLBuffer, weightT: MTLBuffer,
                                       hidden: MTLBuffer, dInput: MTLBuffer, Cin: Int, Cout: Int) {
        precondition(Cout % 4 == 0, "conv1x1_data_grad_relu_t4 requires Cout divisible by 4")
        enc.setComputePipelineState(conv1x1DataGradReluT4PSO)
        enc.setBuffer(dOutput, offset: 0, index: 0)
        enc.setBuffer(weightT, offset: 0, index: 1)
        enc.setBuffer(hidden, offset: 0, index: 2)
        enc.setBuffer(dInput, offset: 0, index: 3)
        var cin = Int32(Cin); enc.setBytes(&cin, length: 4, index: 4)
        var cout = Int32(Cout); enc.setBytes(&cout, length: 4, index: 5)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH * Cin)
    }

    func dispatchFC1DataGradTiled(_ enc: MTLComputeCommandEncoder, dOutput: MTLBuffer,
                                  weightT: MTLBuffer, dInput: MTLBuffer) {
        enc.setComputePipelineState(fc1DataGradTiledPSO)
        enc.setBuffer(dOutput, offset: 0, index: 0)
        enc.setBuffer(weightT, offset: 0, index: 1)
        enc.setBuffer(dInput, offset: 0, index: 2)
        dispatchThreadgroups(
            enc,
            width: (HW + 31) / 32,
            height: (PERC + 7) / 8,
            depth: BATCH,
            threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1)
        )
    }

    func dispatchFC2DataGradReluTiled(_ enc: MTLComputeCommandEncoder, dOutput: MTLBuffer,
                                      weightT: MTLBuffer, hidden: MTLBuffer, dInput: MTLBuffer) {
        enc.setComputePipelineState(fc2DataGradReluTiledPSO)
        enc.setBuffer(dOutput, offset: 0, index: 0)
        enc.setBuffer(weightT, offset: 0, index: 1)
        enc.setBuffer(hidden, offset: 0, index: 2)
        enc.setBuffer(dInput, offset: 0, index: 3)
        dispatchThreadgroups(
            enc,
            width: (HW + 31) / 32,
            height: (HIDDEN + 7) / 8,
            depth: BATCH,
            threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1)
        )
    }

    func dispatchPackChannelsHWMajor(_ enc: MTLComputeCommandEncoder, input: MTLBuffer,
                                     output: MTLBuffer, channels: Int) {
        enc.setComputePipelineState(packChannelsHWMajorPSO)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(output, offset: 0, index: 1)
        var channelCount = Int32(channels)
        enc.setBytes(&channelCount, length: 4, index: 2)
        enc.dispatchThreads(
            MTLSize(width: HW, height: channels, depth: BATCH),
            threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1)
        )
    }

    func dispatchFC1WeightGradTiled(_ enc: MTLComputeCommandEncoder, dOutput: MTLBuffer,
                                    input: MTLBuffer, dWeight: MTLBuffer) {
        enc.setComputePipelineState(fc1WeightGradTiledPSO)
        enc.setBuffer(dOutput, offset: 0, index: 0)
        enc.setBuffer(input, offset: 0, index: 1)
        enc.setBuffer(dWeight, offset: 0, index: 2)
        dispatchThreadgroups(
            enc,
            width: PERC / 16,
            height: HIDDEN,
            depth: 1,
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
    }

    func dispatchFC2WeightGradTiled(_ enc: MTLComputeCommandEncoder, dOutput: MTLBuffer,
                                    input: MTLBuffer, dWeight: MTLBuffer) {
        enc.setComputePipelineState(fc2WeightGradTiledPSO)
        enc.setBuffer(dOutput, offset: 0, index: 0)
        enc.setBuffer(input, offset: 0, index: 1)
        enc.setBuffer(dWeight, offset: 0, index: 2)
        dispatchThreadgroups(
            enc,
            width: HIDDEN / 16,
            height: C,
            depth: 1,
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
    }

    func dispatchConv1x1WeightGrad(_ enc: MTLComputeCommandEncoder, dOutput: MTLBuffer, input: MTLBuffer,
                                   dWeight: MTLBuffer, Cin: Int, Cout: Int) {
        enc.setComputePipelineState(conv1x1WeightGradPSO)
        enc.setBuffer(dOutput, offset: 0, index: 0)
        enc.setBuffer(input, offset: 0, index: 1)
        enc.setBuffer(dWeight, offset: 0, index: 2)
        var cin = Int32(Cin); enc.setBytes(&cin, length: 4, index: 3)
        var cout = Int32(Cout); enc.setBytes(&cout, length: 4, index: 4)
        dispatchThreadgroups(
            enc,
            width: Cin,
            height: Cout,
            depth: 1,
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
    }

    func dispatchBiasGrad(_ enc: MTLComputeCommandEncoder, input: MTLBuffer, output: MTLBuffer) {
        enc.setComputePipelineState(biasGradPSO)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(output, offset: 0, index: 1)
        dispatchThreadgroups(
            enc,
            width: HIDDEN,
            height: 1,
            depth: 1,
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
    }

    // ---- Forward pass (one NCA step) ----
    // Returns all forward intermediates needed for parity checks and backward caching.
    func forwardStep(_ cmd: MTLCommandBuffer, state: MTLBuffer,
                     fc1W: MTLBuffer, fc1B: MTLBuffer, fc2W: MTLBuffer,
                     fireRate: Float = 1.0) -> ForwardStepBuffers {
        let percBuf = makeBuffer(size: BATCH * PERC * HW)
        let fc1OutBuf = makeBuffer(size: BATCH * HIDDEN * HW)
        let hiddenBuf = makeBuffer(size: BATCH * HIDDEN * HW)
        let deltaBuf = makeBuffer(size: BATCH * C * HW)
        let alphaPre = makeBuffer(size: BATCH * HW)
        let maxAlphaPre = makeBuffer(size: BATCH * HW)
        let preMask = makeBuffer(size: BATCH * HW)
        let alphaPost = makeBuffer(size: BATCH * HW)
        let maxAlphaPost = makeBuffer(size: BATCH * HW)
        let postMask = makeBuffer(size: BATCH * HW)
        let lifeMask = makeBuffer(size: BATCH * HW)
        let fireMask = makeBuffer(size: BATCH * HW)
        let updatedBuf = makeBuffer(size: BATCH * C * HW)
        let outputBuf = makeBuffer(size: BATCH * C * HW)

        // Fill fireMask (all 1s for fireRate=1.0, random otherwise)
        let fireMaskPtr = fireMask.contents().bindMemory(to: Float.self, capacity: BATCH * HW)
        if fireRate >= 1.0 {
            for i in 0..<(BATCH * HW) { fireMaskPtr[i] = 1.0 }
        } else {
            for i in 0..<(BATCH * HW) { fireMaskPtr[i] = Float.random(in: 0...1) < fireRate ? 1.0 : 0.0 }
        }

        let onesMask = makeBuffer(size: BATCH * HW)
        let onesPtr = onesMask.contents().bindMemory(to: Float.self, capacity: BATCH * HW)
        for i in 0..<(BATCH * HW) { onesPtr[i] = 1.0 }

        // 1. Perceive
        var enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(perceivePSO)
        enc.setBuffer(state, offset: 0, index: 0)
        enc.setBuffer(percBuf, offset: 0, index: 1)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH * PERC)
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
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH * HIDDEN)
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
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH)
        enc.endEncoding()

        enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(maxPool3x3PSO)
        enc.setBuffer(alphaPre, offset: 0, index: 0)
        enc.setBuffer(maxAlphaPre, offset: 0, index: 1)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH)
        enc.endEncoding()

        enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(thresholdMaskPSO)
        enc.setBuffer(maxAlphaPre, offset: 0, index: 0)
        enc.setBuffer(preMask, offset: 0, index: 1)
        dispatch1D(enc, count: BATCH * HW)
        enc.endEncoding()

        // 6. Compute updated = state + delta * fireMask with a unit life mask.
        enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(maskedResidualPSO)
        enc.setBuffer(state, offset: 0, index: 0)
        enc.setBuffer(deltaBuf, offset: 0, index: 1)
        enc.setBuffer(fireMask, offset: 0, index: 2)
        enc.setBuffer(onesMask, offset: 0, index: 3)
        enc.setBuffer(updatedBuf, offset: 0, index: 4)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH * C)
        enc.endEncoding()

        // 7. Post-alive mask on updated.
        enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(sliceAlphaPSO)
        enc.setBuffer(updatedBuf, offset: 0, index: 0)
        enc.setBuffer(alphaPost, offset: 0, index: 1)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH)
        enc.endEncoding()

        enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(maxPool3x3PSO)
        enc.setBuffer(alphaPost, offset: 0, index: 0)
        enc.setBuffer(maxAlphaPost, offset: 0, index: 1)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH)
        enc.endEncoding()

        enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(thresholdMaskPSO)
        enc.setBuffer(maxAlphaPost, offset: 0, index: 0)
        enc.setBuffer(postMask, offset: 0, index: 1)
        dispatch1D(enc, count: BATCH * HW)
        enc.endEncoding()

        // 8. lifeMask = preMask * postMask.
        enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(maskMultiplyPSO)
        enc.setBuffer(preMask, offset: 0, index: 0)
        enc.setBuffer(postMask, offset: 0, index: 1)
        enc.setBuffer(lifeMask, offset: 0, index: 2)
        dispatch1D(enc, count: BATCH * HW)
        enc.endEncoding()

        // 9. final output = updated * lifeMask.
        enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(applyLifeMaskPSO)
        enc.setBuffer(updatedBuf, offset: 0, index: 0)
        enc.setBuffer(lifeMask, offset: 0, index: 1)
        enc.setBuffer(outputBuf, offset: 0, index: 2)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH * C)
        enc.endEncoding()

        return ForwardStepBuffers(
            perc: percBuf,
            fc1Out: fc1OutBuf,
            hidden: hiddenBuf,
            delta: deltaBuf,
            maxAlphaPre: maxAlphaPre,
            preMask: preMask,
            updated: updatedBuf,
            maxAlphaPost: maxAlphaPost,
            postMask: postMask,
            lifeMask: lifeMask,
            fireMask: fireMask,
            output: outputBuf
        )
    }

    func backwardStep(_ cmd: MTLCommandBuffer, dOutput: MTLBuffer, forward: ForwardStepBuffers,
                      fc1W: MTLBuffer, fc2W: MTLBuffer) -> BackwardStepBuffers {
        let dUpdated = makeBuffer(size: BATCH * C * HW)
        let dDelta = makeBuffer(size: BATCH * C * HW)
        let dHidden = makeBuffer(size: BATCH * HIDDEN * HW)
        let dFC2W = makeBuffer(size: C * HIDDEN)
        let dFC1Raw = makeBuffer(size: BATCH * HIDDEN * HW)
        let dFC1B = makeBuffer(size: HIDDEN)
        let dPerc = makeBuffer(size: BATCH * PERC * HW)
        let dFC1W = makeBuffer(size: HIDDEN * PERC)
        let dStateFromPerc = makeBuffer(size: BATCH * C * HW)
        let dState = makeBuffer(size: BATCH * C * HW)

        var enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(bwdMaskedResidualPSO)
        enc.setBuffer(dOutput, offset: 0, index: 0)
        enc.setBuffer(forward.fireMask, offset: 0, index: 1)
        enc.setBuffer(forward.lifeMask, offset: 0, index: 2)
        enc.setBuffer(dUpdated, offset: 0, index: 3)
        enc.setBuffer(dDelta, offset: 0, index: 4)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH * C)
        enc.endEncoding()

        enc = cmd.makeComputeCommandEncoder()!
        dispatchConv1x1DataGrad(enc, dOutput: dDelta, weight: fc2W, dInput: dHidden, Cin: HIDDEN, Cout: C)
        enc.endEncoding()

        enc = cmd.makeComputeCommandEncoder()!
        dispatchConv1x1WeightGrad(enc, dOutput: dDelta, input: forward.hidden, dWeight: dFC2W, Cin: HIDDEN, Cout: C)
        enc.endEncoding()

        enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(reluBackwardPSO)
        enc.setBuffer(dHidden, offset: 0, index: 0)
        enc.setBuffer(forward.hidden, offset: 0, index: 1)
        enc.setBuffer(dFC1Raw, offset: 0, index: 2)
        dispatch1D(enc, count: BATCH * HIDDEN * HW)
        enc.endEncoding()

        enc = cmd.makeComputeCommandEncoder()!
        dispatchBiasGrad(enc, input: dFC1Raw, output: dFC1B)
        enc.endEncoding()

        enc = cmd.makeComputeCommandEncoder()!
        dispatchConv1x1DataGrad(enc, dOutput: dFC1Raw, weight: fc1W, dInput: dPerc, Cin: PERC, Cout: HIDDEN)
        enc.endEncoding()

        enc = cmd.makeComputeCommandEncoder()!
        dispatchConv1x1WeightGrad(enc, dOutput: dFC1Raw, input: forward.perc, dWeight: dFC1W, Cin: PERC, Cout: HIDDEN)
        enc.endEncoding()

        enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(perceiveBackwardPSO)
        enc.setBuffer(dPerc, offset: 0, index: 0)
        enc.setBuffer(dStateFromPerc, offset: 0, index: 1)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH * C)
        enc.endEncoding()

        enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(elemAddPSO)
        enc.setBuffer(dUpdated, offset: 0, index: 0)
        enc.setBuffer(dStateFromPerc, offset: 0, index: 1)
        enc.setBuffer(dState, offset: 0, index: 2)
        dispatch1D(enc, count: BATCH * C * HW)
        enc.endEncoding()

        return BackwardStepBuffers(
            dUpdated: dUpdated,
            dDelta: dDelta,
            dHidden: dHidden,
            dFC2W: dFC2W,
            dFC1Raw: dFC1Raw,
            dFC1B: dFC1B,
            dPerc: dPerc,
            dFC1W: dFC1W,
            dStateFromPerc: dStateFromPerc,
            dState: dState
        )
    }

    func makeRolloutScratch(options: MTLResourceOptions = .storageModeShared) -> RolloutScratch {
        return RolloutScratch(
            perc: makeBuffer(size: BATCH * PERC * HW, options: options),
            hidden: makeBuffer(size: BATCH * HIDDEN * HW, options: options),
            delta: makeBuffer(size: BATCH * C * HW, options: options),
            preMask: makeBuffer(size: BATCH * HW, options: options),
            postMask: makeBuffer(size: BATCH * HW, options: options)
        )
    }

    func encodeForwardStepFast(_ enc: MTLComputeCommandEncoder, stateIn: MTLBuffer, stateOut: MTLBuffer,
                               scratch: RolloutScratch, fc1WT: MTLBuffer, fc1B: MTLBuffer, fc2WT: MTLBuffer) {
        enc.setComputePipelineState(perceiveX3PSO)
        enc.setBuffer(stateIn, offset: 0, index: 0)
        enc.setBuffer(scratch.perc, offset: 0, index: 1)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH * C)

        dispatchConv1x1FwdX8(enc, input: scratch.perc, weightT: fc1WT, output: scratch.hidden, Cin: PERC, Cout: HIDDEN)

        enc.setComputePipelineState(biasReluPSO)
        enc.setBuffer(scratch.hidden, offset: 0, index: 0)
        enc.setBuffer(fc1B, offset: 0, index: 1)
        enc.setBuffer(scratch.hidden, offset: 0, index: 2)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH * HIDDEN)

        dispatchConv1x1FwdX8(enc, input: scratch.hidden, weightT: fc2WT, output: scratch.delta, Cin: HIDDEN, Cout: C)

        enc.setComputePipelineState(aliveMaskFromStatePSO)
        enc.setBuffer(stateIn, offset: 0, index: 0)
        enc.setBuffer(scratch.preMask, offset: 0, index: 1)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH)

        enc.setComputePipelineState(residualAddPSO)
        enc.setBuffer(stateIn, offset: 0, index: 0)
        enc.setBuffer(scratch.delta, offset: 0, index: 1)
        enc.setBuffer(stateOut, offset: 0, index: 2)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH * C)

        enc.setComputePipelineState(aliveMaskFromStatePSO)
        enc.setBuffer(stateOut, offset: 0, index: 0)
        enc.setBuffer(scratch.postMask, offset: 0, index: 1)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH)

        enc.setComputePipelineState(applyTwoMasksPSO)
        enc.setBuffer(stateOut, offset: 0, index: 0)
        enc.setBuffer(scratch.preMask, offset: 0, index: 1)
        enc.setBuffer(scratch.postMask, offset: 0, index: 2)
        enc.setBuffer(stateOut, offset: 0, index: 3)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH * C)
    }

    func encodeForwardTwoStepsFused(_ enc: MTLComputeCommandEncoder, stateIn: MTLBuffer, stateOut: MTLBuffer,
                                    fc1WT: MTLBuffer, fc1B: MTLBuffer, fc2WT: MTLBuffer) {
        enc.setComputePipelineState(rolloutTwoStepsFusedPSO)
        enc.setBuffer(stateIn, offset: 0, index: 0)
        enc.setBuffer(fc1WT, offset: 0, index: 1)
        enc.setBuffer(fc1B, offset: 0, index: 2)
        enc.setBuffer(fc2WT, offset: 0, index: 3)
        enc.setBuffer(stateOut, offset: 0, index: 4)
        dispatchThreadgroups(
            enc,
            width: GS / 8,
            height: GS / 8,
            depth: BATCH,
            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1)
        )
    }

    @discardableResult
    func encodeForwardRolloutFast(_ enc: MTLComputeCommandEncoder, steps: Int,
                                  stateA: MTLBuffer, stateB: MTLBuffer, scratch: RolloutScratch,
                                  fc1WT: MTLBuffer, fc1B: MTLBuffer, fc2WT: MTLBuffer) -> MTLBuffer {
        var currentIn = stateA
        var currentOut = stateB
        for _ in 0..<steps {
            encodeForwardStepFast(enc, stateIn: currentIn, stateOut: currentOut, scratch: scratch,
                                  fc1WT: fc1WT, fc1B: fc1B, fc2WT: fc2WT)
            swap(&currentIn, &currentOut)
        }
        return currentIn
    }

    @discardableResult
    func encodeForwardRolloutFused2(_ enc: MTLComputeCommandEncoder, steps: Int,
                                    stateA: MTLBuffer, stateB: MTLBuffer, scratch: RolloutScratch,
                                    fc1WT: MTLBuffer, fc1B: MTLBuffer, fc2WT: MTLBuffer) -> MTLBuffer {
        var currentIn = stateA
        var currentOut = stateB
        var remaining = steps

        while remaining >= 2 {
            encodeForwardTwoStepsFused(enc, stateIn: currentIn, stateOut: currentOut,
                                       fc1WT: fc1WT, fc1B: fc1B, fc2WT: fc2WT)
            swap(&currentIn, &currentOut)
            remaining -= 2
        }

        if remaining == 1 {
            encodeForwardStepFast(enc, stateIn: currentIn, stateOut: currentOut, scratch: scratch,
                                  fc1WT: fc1WT, fc1B: fc1B, fc2WT: fc2WT)
            swap(&currentIn, &currentOut)
        }

        return currentIn
    }

    func makeFastTrainScratch(options: MTLResourceOptions = .storageModeShared) -> FastTrainScratch {
        FastTrainScratch(
            perc: makeBuffer(size: BATCH * PERC * HW, options: options),
            percHWMajor: makeBuffer(size: BATCH * HW * PERC, options: options),
            hidden: makeBuffer(size: BATCH * HIDDEN * HW, options: options),
            hiddenHWMajor: makeBuffer(size: BATCH * HW * HIDDEN, options: options),
            delta: makeBuffer(size: BATCH * C * HW, options: options),
            preMask: makeBuffer(size: BATCH * HW, options: options),
            postMask: makeBuffer(size: BATCH * HW, options: options),
            lifeMask: makeBuffer(size: BATCH * HW, options: options),
            output: makeBuffer(size: BATCH * C * HW, options: options),
            dResidual: makeBuffer(size: BATCH * C * HW, options: options),
            dFC2W: makeBuffer(size: C * HIDDEN, options: options),
            dFC1Raw: makeBuffer(size: BATCH * HIDDEN * HW, options: options),
            dFC1B: makeBuffer(size: HIDDEN, options: options),
            dPerc: makeBuffer(size: BATCH * PERC * HW, options: options),
            dFC1W: makeBuffer(size: HIDDEN * PERC, options: options),
            dStateFromPerc: makeBuffer(size: BATCH * C * HW, options: options),
            dState: makeBuffer(size: BATCH * C * HW, options: options)
        )
    }

    func encodeForwardStepTrainFast(_ enc: MTLComputeCommandEncoder, stateIn: MTLBuffer, scratch: FastTrainScratch,
                                    fc1WT: MTLBuffer, fc1B: MTLBuffer, fc2WT: MTLBuffer) {
        enc.setComputePipelineState(perceiveX3PSO)
        enc.setBuffer(stateIn, offset: 0, index: 0)
        enc.setBuffer(scratch.perc, offset: 0, index: 1)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH * C)

        dispatchConv1x1FwdX8(enc, input: scratch.perc, weightT: fc1WT, output: scratch.hidden, Cin: PERC, Cout: HIDDEN)

        enc.setComputePipelineState(biasReluPSO)
        enc.setBuffer(scratch.hidden, offset: 0, index: 0)
        enc.setBuffer(fc1B, offset: 0, index: 1)
        enc.setBuffer(scratch.hidden, offset: 0, index: 2)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH * HIDDEN)

        dispatchConv1x1FwdX8(enc, input: scratch.hidden, weightT: fc2WT, output: scratch.delta, Cin: HIDDEN, Cout: C)

        enc.setComputePipelineState(aliveMaskFromStatePSO)
        enc.setBuffer(stateIn, offset: 0, index: 0)
        enc.setBuffer(scratch.preMask, offset: 0, index: 1)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH)

        enc.setComputePipelineState(residualAddPSO)
        enc.setBuffer(stateIn, offset: 0, index: 0)
        enc.setBuffer(scratch.delta, offset: 0, index: 1)
        enc.setBuffer(scratch.output, offset: 0, index: 2)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH * C)

        enc.setComputePipelineState(aliveMaskFromStatePSO)
        enc.setBuffer(scratch.output, offset: 0, index: 0)
        enc.setBuffer(scratch.postMask, offset: 0, index: 1)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH)

        enc.setComputePipelineState(applyTwoMasksStoreLifePSO)
        enc.setBuffer(scratch.output, offset: 0, index: 0)
        enc.setBuffer(scratch.preMask, offset: 0, index: 1)
        enc.setBuffer(scratch.postMask, offset: 0, index: 2)
        enc.setBuffer(scratch.output, offset: 0, index: 3)
        enc.setBuffer(scratch.lifeMask, offset: 0, index: 4)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH * C)
    }

    func encodeBackwardStepFast(_ enc: MTLComputeCommandEncoder, dOutput: MTLBuffer, scratch: FastTrainScratch,
                                fc1WT: MTLBuffer, fc2WT: MTLBuffer) {
        enc.setComputePipelineState(applyLifeMaskPSO)
        enc.setBuffer(dOutput, offset: 0, index: 0)
        enc.setBuffer(scratch.lifeMask, offset: 0, index: 1)
        enc.setBuffer(scratch.dResidual, offset: 0, index: 2)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH * C)

        dispatchFC2WeightGradTiled(enc, dOutput: scratch.dResidual, input: scratch.hidden, dWeight: scratch.dFC2W)
        dispatchFC2DataGradReluTiled(enc, dOutput: scratch.dResidual, weightT: fc2WT, hidden: scratch.hidden, dInput: scratch.dFC1Raw)
        dispatchBiasGrad(enc, input: scratch.dFC1Raw, output: scratch.dFC1B)
        dispatchFC1WeightGradTiled(enc, dOutput: scratch.dFC1Raw, input: scratch.perc, dWeight: scratch.dFC1W)
        dispatchFC1DataGradTiled(enc, dOutput: scratch.dFC1Raw, weightT: fc1WT, dInput: scratch.dPerc)

        enc.setComputePipelineState(perceiveBackwardPSO)
        enc.setBuffer(scratch.dPerc, offset: 0, index: 0)
        enc.setBuffer(scratch.dStateFromPerc, offset: 0, index: 1)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH * C)

        enc.setComputePipelineState(elemAddPSO)
        enc.setBuffer(scratch.dResidual, offset: 0, index: 0)
        enc.setBuffer(scratch.dStateFromPerc, offset: 0, index: 1)
        enc.setBuffer(scratch.dState, offset: 0, index: 2)
        dispatch1D(enc, count: BATCH * C * HW)
    }

    func encodeBackwardCoreFast(_ enc: MTLComputeCommandEncoder, dOutput: MTLBuffer, scratch: FastTrainScratch,
                                fc1WT: MTLBuffer, fc2WT: MTLBuffer) {
        enc.setComputePipelineState(applyLifeMaskPSO)
        enc.setBuffer(dOutput, offset: 0, index: 0)
        enc.setBuffer(scratch.lifeMask, offset: 0, index: 1)
        enc.setBuffer(scratch.dResidual, offset: 0, index: 2)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH * C)

        dispatchFC2DataGradReluTiled(enc, dOutput: scratch.dResidual, weightT: fc2WT, hidden: scratch.hidden, dInput: scratch.dFC1Raw)
        dispatchBiasGrad(enc, input: scratch.dFC1Raw, output: scratch.dFC1B)
        dispatchFC1DataGradTiled(enc, dOutput: scratch.dFC1Raw, weightT: fc1WT, dInput: scratch.dPerc)

        enc.setComputePipelineState(perceiveBackwardPSO)
        enc.setBuffer(scratch.dPerc, offset: 0, index: 0)
        enc.setBuffer(scratch.dStateFromPerc, offset: 0, index: 1)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH * C)

        enc.setComputePipelineState(elemAddPSO)
        enc.setBuffer(scratch.dResidual, offset: 0, index: 0)
        enc.setBuffer(scratch.dStateFromPerc, offset: 0, index: 1)
        enc.setBuffer(scratch.dState, offset: 0, index: 2)
        dispatch1D(enc, count: BATCH * C * HW)
    }

    func encodeBackwardWeightGradFast(_ enc: MTLComputeCommandEncoder, scratch: FastTrainScratch) {
        dispatchFC2WeightGradTiled(enc, dOutput: scratch.dResidual, input: scratch.hidden, dWeight: scratch.dFC2W)
        dispatchFC1WeightGradTiled(enc, dOutput: scratch.dFC1Raw, input: scratch.perc, dWeight: scratch.dFC1W)
    }

    func encodeBackwardWeightGradPrepFast(_ enc: MTLComputeCommandEncoder, dOutput: MTLBuffer, scratch: FastTrainScratch,
                                          fc2WT: MTLBuffer) {
        enc.setComputePipelineState(applyLifeMaskPSO)
        enc.setBuffer(dOutput, offset: 0, index: 0)
        enc.setBuffer(scratch.lifeMask, offset: 0, index: 1)
        enc.setBuffer(scratch.dResidual, offset: 0, index: 2)
        dispatchGrid(enc, width: GS, height: GS, depth: BATCH * C)

        dispatchFC2DataGradReluTiled(enc, dOutput: scratch.dResidual, weightT: fc2WT, hidden: scratch.hidden, dInput: scratch.dFC1Raw)
    }
}

// MARK: - Verification / Benchmark Harness
struct LoadedWeights {
    let fc1WData: [Float]
    let fc1BData: [Float]
    let fc2WData: [Float]
    let fc1WTData: [Float]
    let fc2WTData: [Float]
    let fc1W: MTLBuffer
    let fc1B: MTLBuffer
    let fc2W: MTLBuffer
    let fc1WT: MTLBuffer
    let fc2WT: MTLBuffer
}

enum RunMode: String {
    case accuracy
    case benchmark
    case all
}

struct RunOptions {
    var mode: RunMode = .all
    var rolloutSteps: Int = 10
    var benchmarkIterations: Int = 20
    var warmupIterations: Int = 3
}

struct BenchmarkSummary {
    let label: String
    let averageCPUms: Double
    let minCPUms: Double
    let maxCPUms: Double
    let averageGPUms: Double?
    let workItemsPerIteration: Int
}

func printUsage() {
    print("""
    Usage: NCATrainer [accuracy|benchmark|all] [--rollout-steps N] [--benchmark-iterations N] [--warmup N]
    """)
}

func parseOptions() -> RunOptions {
    var options = RunOptions()
    let args = Array(CommandLine.arguments.dropFirst())
    var idx = 0

    while idx < args.count {
        let arg = args[idx]
        switch arg {
        case "accuracy":
            options.mode = .accuracy
        case "benchmark":
            options.mode = .benchmark
        case "all":
            options.mode = .all
        case "--rollout-steps":
            idx += 1
            options.rolloutSteps = Int(args[idx])!
        case "--benchmark-iterations":
            idx += 1
            options.benchmarkIterations = Int(args[idx])!
        case "--warmup":
            idx += 1
            options.warmupIterations = Int(args[idx])!
        case "--help", "-h":
            printUsage()
            exit(0)
        default:
            fatalError("Unknown argument: \(arg)")
        }
        idx += 1
    }

    return options
}

func loadBinaryFloats(_ path: String) -> [Float] {
    let data = try! Data(contentsOf: URL(fileURLWithPath: path))
    return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

func loadOpRef(_ name: String) -> [Float] {
    loadBinaryFloats("reference_ops/\(name).bin")
}

func loadRolloutRef(step: Int) -> [Float] {
    loadBinaryFloats(String(format: "reference/step_%03d.bin", step))
}

func transposeWeights(_ weight: [Float], cin: Int, cout: Int) -> [Float] {
    var transposed = [Float](repeating: 0, count: weight.count)
    for co in 0..<cout {
        for ci in 0..<cin {
            transposed[ci * cout + co] = weight[co * cin + ci]
        }
    }
    return transposed
}

func loadWeights(_ metal: NCAMetal) -> LoadedWeights {
    let allW = loadBinaryFloats("../output/A/weights.bin")
    let fc1WData = Array(allW[0..<(HIDDEN * PERC)])
    let fc1BData = Array(allW[(HIDDEN * PERC)..<(HIDDEN * PERC + HIDDEN)])
    let fc2WData = Array(allW[(HIDDEN * PERC + HIDDEN)..<(HIDDEN * PERC + HIDDEN + C * HIDDEN)])
    let fc1WTData = transposeWeights(fc1WData, cin: PERC, cout: HIDDEN)
    let fc2WTData = transposeWeights(fc2WData, cin: HIDDEN, cout: C)
    return LoadedWeights(
        fc1WData: fc1WData,
        fc1BData: fc1BData,
        fc2WData: fc2WData,
        fc1WTData: fc1WTData,
        fc2WTData: fc2WTData,
        fc1W: metal.makePrivateBuffer(fc1WData),
        fc1B: metal.makePrivateBuffer(fc1BData),
        fc2W: metal.makePrivateBuffer(fc2WData),
        fc1WT: metal.makePrivateBuffer(fc1WTData),
        fc2WT: metal.makePrivateBuffer(fc2WTData)
    )
}

func repeatSample(_ sample: [Float], count: Int) -> [Float] {
    var repeated = [Float]()
    repeated.reserveCapacity(sample.count * count)
    for _ in 0..<count {
        repeated.append(contentsOf: sample)
    }
    return repeated
}

func sample0(_ batched: [Float], sampleSize: Int) -> [Float] {
    Array(batched[0..<sampleSize])
}

@discardableResult
func compare(_ name: String, _ gpu: [Float], _ ref: [Float], tolerance: Float = 1e-4) -> Bool {
    guard gpu.count == ref.count else {
        print("  \(name): SIZE MISMATCH \(gpu.count) vs \(ref.count)")
        return false
    }

    var maxErr: Float = 0
    var maxIdx = 0
    for i in 0..<gpu.count {
        let e = abs(gpu[i] - ref[i])
        if e > maxErr { maxErr = e; maxIdx = i }
    }

    let passed = maxErr <= tolerance
    let padded = name.padding(toLength: 15, withPad: " ", startingAt: 0)
    print("  \(padded) max_err=\(String(format: "%.3e", maxErr)) idx=\(maxIdx) \(passed ? "PASS" : "FAIL")")
    return passed
}

func addChannelBias(_ tensor: [Float], bias: [Float], channels: Int) -> [Float] {
    var result = tensor
    for b in 0..<BATCH {
        for c in 0..<channels {
            let base = (b * channels + c) * HW
            let biasValue = bias[c]
            for offset in 0..<HW {
                result[base + offset] += biasValue
            }
        }
    }
    return result
}

func commitAndWait(_ cmd: MTLCommandBuffer, label: String) {
    log("Committing \(label)...")
    cmd.commit()
    cmd.waitUntilCompleted()
    if let error = cmd.error {
        fatalError("\(label) failed: \(error)")
    }
}

func verifySingleStep(_ metal: NCAMetal, _ weights: LoadedWeights) -> Bool {
    var ok = true

    print("\n=== Single-Step Forward Accuracy ===")
    let stateBuf = metal.makeBuffer(loadOpRef("input"))
    let fwdCmd = metal.queue.makeCommandBuffer()!
    let forward = metal.forwardStep(fwdCmd, state: stateBuf, fc1W: weights.fc1W, fc1B: weights.fc1B, fc2W: weights.fc2W, fireRate: 1.0)
    commitAndWait(fwdCmd, label: "forward command buffer")

    let percOut = metal.readBuffer(forward.perc, count: BATCH * PERC * HW)
    let fc1Out = metal.readBuffer(forward.fc1Out, count: BATCH * HIDDEN * HW)
    let hiddenOut = metal.readBuffer(forward.hidden, count: BATCH * HIDDEN * HW)
    let deltaOut = metal.readBuffer(forward.delta, count: BATCH * C * HW)
    let maxAlphaPreOut = metal.readBuffer(forward.maxAlphaPre, count: BATCH * HW)
    let preMaskOut = metal.readBuffer(forward.preMask, count: BATCH * HW)
    let updatedOut = metal.readBuffer(forward.updated, count: BATCH * C * HW)
    let maxAlphaPostOut = metal.readBuffer(forward.maxAlphaPost, count: BATCH * HW)
    let postMaskOut = metal.readBuffer(forward.postMask, count: BATCH * HW)
    let lifeMaskOut = metal.readBuffer(forward.lifeMask, count: BATCH * HW)
    let outputOut = metal.readBuffer(forward.output, count: BATCH * C * HW)

    ok = compare("perc", percOut, loadOpRef("perc")) && ok
    ok = compare("fc1_biased", addChannelBias(fc1Out, bias: weights.fc1BData, channels: HIDDEN), loadOpRef("fc1_out")) && ok
    ok = compare("hidden", hiddenOut, loadOpRef("hidden")) && ok
    ok = compare("delta", deltaOut, loadOpRef("delta")) && ok
    ok = compare("max_alpha_pre", maxAlphaPreOut, loadOpRef("max_alpha_pre")) && ok
    ok = compare("pre_mask", preMaskOut, loadOpRef("pre_mask")) && ok
    ok = compare("updated", updatedOut, loadOpRef("updated")) && ok
    ok = compare("max_alpha_post", maxAlphaPostOut, loadOpRef("max_alpha_post")) && ok
    ok = compare("post_mask", postMaskOut, loadOpRef("post_mask")) && ok
    ok = compare("life_mask", lifeMaskOut, loadOpRef("life_mask")) && ok
    ok = compare("output", outputOut, loadOpRef("output")) && ok

    print("\n=== Single-Step Backward Accuracy ===")
    let dOutputBuf = metal.makeBuffer(loadOpRef("d_output"))
    let bwdCmd = metal.queue.makeCommandBuffer()!
    let backward = metal.backwardStep(bwdCmd, dOutput: dOutputBuf, forward: forward, fc1W: weights.fc1W, fc2W: weights.fc2W)
    commitAndWait(bwdCmd, label: "backward command buffer")

    ok = compare("d_updated", metal.readBuffer(backward.dUpdated, count: BATCH * C * HW), loadOpRef("d_updated")) && ok
    ok = compare("d_delta", metal.readBuffer(backward.dDelta, count: BATCH * C * HW), loadOpRef("d_delta")) && ok
    ok = compare("d_hidden", metal.readBuffer(backward.dHidden, count: BATCH * HIDDEN * HW), loadOpRef("d_hidden")) && ok
    ok = compare("d_fc2_w", metal.readBuffer(backward.dFC2W, count: C * HIDDEN), loadOpRef("d_fc2_w"), tolerance: 5e-4) && ok
    ok = compare("d_fc1_raw", metal.readBuffer(backward.dFC1Raw, count: BATCH * HIDDEN * HW), loadOpRef("d_fc1_raw")) && ok
    ok = compare("d_fc1_b", metal.readBuffer(backward.dFC1B, count: HIDDEN), loadOpRef("d_fc1_b"), tolerance: 5e-4) && ok
    ok = compare("d_perc", metal.readBuffer(backward.dPerc, count: BATCH * PERC * HW), loadOpRef("d_perc")) && ok
    ok = compare("d_fc1_w", metal.readBuffer(backward.dFC1W, count: HIDDEN * PERC), loadOpRef("d_fc1_w"), tolerance: 5e-4) && ok
    ok = compare("d_state", metal.readBuffer(backward.dState, count: BATCH * C * HW), loadOpRef("d_state"), tolerance: 5e-4) && ok

    return ok
}

func verifyRollout(_ metal: NCAMetal, _ weights: LoadedWeights, steps: Int) -> Bool {
    print("\n=== GPU-Resident Rollout Final Accuracy (\(steps) steps) ===")
    let seedBatch = repeatSample(loadRolloutRef(step: 0), count: BATCH)
    let seedPrivate = metal.makePrivateBuffer(seedBatch)
    let stateA = metal.makeBuffer(size: BATCH * STATE, options: .storageModePrivate)
    let stateB = metal.makeBuffer(size: BATCH * STATE, options: .storageModePrivate)
    let scratch = metal.makeRolloutScratch(options: .storageModePrivate)
    let finalShared = metal.makeBuffer(size: BATCH * STATE)
    let cmd = metal.queue.makeCommandBuffer()!
    let seedBlit = cmd.makeBlitCommandEncoder()!
    seedBlit.copy(from: seedPrivate, sourceOffset: 0, to: stateA, destinationOffset: 0, size: seedPrivate.length)
    seedBlit.endEncoding()

    let enc = cmd.makeComputeCommandEncoder()!
    let finalPrivate = metal.encodeForwardRolloutFast(enc, steps: steps, stateA: stateA, stateB: stateB, scratch: scratch,
                                                      fc1WT: weights.fc1WT, fc1B: weights.fc1B, fc2WT: weights.fc2WT)
    enc.endEncoding()

    let readbackBlit = cmd.makeBlitCommandEncoder()!
    readbackBlit.copy(from: finalPrivate, sourceOffset: 0, to: finalShared, destinationOffset: 0, size: finalShared.length)
    readbackBlit.endEncoding()

    commitAndWait(cmd, label: "gpu-resident rollout command buffer")
    let batched = metal.readBuffer(finalShared, count: BATCH * STATE)
    return compare("rollout_final", sample0(batched, sampleSize: STATE), loadRolloutRef(step: steps), tolerance: 5e-4)
}

func average(_ values: [Double]) -> Double {
    values.reduce(0, +) / Double(values.count)
}

func benchmark(_ label: String, iterations: Int, warmup: Int, workItemsPerIteration: Int,
               build: () -> MTLCommandBuffer) -> BenchmarkSummary {
    var cpuMs = [Double]()
    var gpuMs = [Double]()

    for iter in 0..<(warmup + iterations) {
        let startNs = DispatchTime.now().uptimeNanoseconds
        let cmd = build()
        cmd.commit()
        cmd.waitUntilCompleted()
        if let error = cmd.error {
            fatalError("\(label) benchmark failed: \(error)")
        }

        let elapsedMs = Double(DispatchTime.now().uptimeNanoseconds - startNs) / 1_000_000.0
        if iter >= warmup {
            cpuMs.append(elapsedMs)
            if cmd.gpuEndTime > cmd.gpuStartTime {
                gpuMs.append((cmd.gpuEndTime - cmd.gpuStartTime) * 1000.0)
            }
        }
    }

    return BenchmarkSummary(
        label: label,
        averageCPUms: average(cpuMs),
        minCPUms: cpuMs.min() ?? 0,
        maxCPUms: cpuMs.max() ?? 0,
        averageGPUms: gpuMs.isEmpty ? nil : average(gpuMs),
        workItemsPerIteration: workItemsPerIteration
    )
}

func printBenchmark(_ summary: BenchmarkSummary) {
    let throughput = Double(summary.workItemsPerIteration) * 1000.0 / summary.averageCPUms
    if let gpuMs = summary.averageGPUms {
        print("  \(summary.label): cpu_avg=\(String(format: "%.2f", summary.averageCPUms))ms gpu_avg=\(String(format: "%.2f", gpuMs))ms throughput=\(String(format: "%.2f", throughput))/s min=\(String(format: "%.2f", summary.minCPUms))ms max=\(String(format: "%.2f", summary.maxCPUms))ms")
    } else {
        print("  \(summary.label): cpu_avg=\(String(format: "%.2f", summary.averageCPUms))ms throughput=\(String(format: "%.2f", throughput))/s min=\(String(format: "%.2f", summary.minCPUms))ms max=\(String(format: "%.2f", summary.maxCPUms))ms")
    }
}

func runBenchmarks(_ metal: NCAMetal, _ weights: LoadedWeights, options: RunOptions) {
    print("\n=== Benchmarks ===")
    let seedBatch = repeatSample(loadRolloutRef(step: 0), count: BATCH)
    let opInputShared = metal.makeBuffer(loadOpRef("input"))
    let dOutputShared = metal.makeBuffer(loadOpRef("d_output"))
    let opInputPrivate = metal.makePrivateCopy(of: opInputShared)
    let dOutputPrivate = metal.makePrivateCopy(of: dOutputShared)
    let rolloutSeedPrivate = metal.makePrivateBuffer(seedBatch)
    let rolloutStateA = metal.makeBuffer(size: BATCH * STATE, options: .storageModePrivate)
    let rolloutStateB = metal.makeBuffer(size: BATCH * STATE, options: .storageModePrivate)
    let rolloutScratch = metal.makeRolloutScratch(options: .storageModePrivate)
    let rolloutReadback = metal.makeBuffer(size: BATCH * STATE)
    let fastTrainValidationScratch = metal.makeFastTrainScratch()
    let fastTrainBenchmarkScratch = metal.makeFastTrainScratch(options: .storageModePrivate)

    // Validate the GPU-resident rollout path against the PyTorch reference before timing it.
    let validationCmd = metal.queue.makeCommandBuffer()!
    let validationSeedBlit = validationCmd.makeBlitCommandEncoder()!
    validationSeedBlit.copy(from: rolloutSeedPrivate, sourceOffset: 0, to: rolloutStateA, destinationOffset: 0, size: rolloutSeedPrivate.length)
    validationSeedBlit.endEncoding()
    let validationEnc = validationCmd.makeComputeCommandEncoder()!
    let fastFinalBuf = metal.encodeForwardRolloutFast(validationEnc, steps: options.rolloutSteps,
                                                      stateA: rolloutStateA, stateB: rolloutStateB, scratch: rolloutScratch,
                                                      fc1WT: weights.fc1WT, fc1B: weights.fc1B, fc2WT: weights.fc2WT)
    validationEnc.endEncoding()
    let validationReadbackBlit = validationCmd.makeBlitCommandEncoder()!
    validationReadbackBlit.copy(from: fastFinalBuf, sourceOffset: 0, to: rolloutReadback, destinationOffset: 0, size: rolloutReadback.length)
    validationReadbackBlit.endEncoding()
    commitAndWait(validationCmd, label: "fast rollout validation")
    let fastFinal = metal.readBuffer(rolloutReadback, count: BATCH * STATE)
    guard compare("fast_final", sample0(fastFinal, sampleSize: STATE), loadRolloutRef(step: options.rolloutSteps), tolerance: 5e-4) else {
        fatalError("Fast rollout path mismatched the reference output")
    }

    let fusedValidationCmd = metal.queue.makeCommandBuffer()!
    let fusedValidationSeedBlit = fusedValidationCmd.makeBlitCommandEncoder()!
    fusedValidationSeedBlit.copy(from: rolloutSeedPrivate, sourceOffset: 0, to: rolloutStateA, destinationOffset: 0, size: rolloutSeedPrivate.length)
    fusedValidationSeedBlit.endEncoding()
    let fusedValidationEnc = fusedValidationCmd.makeComputeCommandEncoder()!
    let fusedFinalBuf = metal.encodeForwardRolloutFused2(fusedValidationEnc, steps: options.rolloutSteps,
                                                         stateA: rolloutStateA, stateB: rolloutStateB, scratch: rolloutScratch,
                                                         fc1WT: weights.fc1WT, fc1B: weights.fc1B, fc2WT: weights.fc2WT)
    fusedValidationEnc.endEncoding()
    let fusedReadbackBlit = fusedValidationCmd.makeBlitCommandEncoder()!
    fusedReadbackBlit.copy(from: fusedFinalBuf, sourceOffset: 0, to: rolloutReadback, destinationOffset: 0, size: rolloutReadback.length)
    fusedReadbackBlit.endEncoding()
    commitAndWait(fusedValidationCmd, label: "fused rollout validation")
    let fusedFinal = metal.readBuffer(rolloutReadback, count: BATCH * STATE)
    let fusedSample = sample0(fusedFinal, sampleSize: STATE)
    let fusedRef = loadRolloutRef(step: options.rolloutSteps)
    if !compare("fused2_final", fusedSample, fusedRef, tolerance: 5e-4) {
        var maxErr: Float = 0
        var maxIdx = 0
        for i in 0..<fusedSample.count {
            let err = abs(fusedSample[i] - fusedRef[i])
            if err > maxErr {
                maxErr = err
                maxIdx = i
            }
        }
        log("Fused 2-step rollout mismatch: max_err=\(maxErr) idx=\(maxIdx) gpu=\(fusedSample[maxIdx]) ref=\(fusedRef[maxIdx])")
    }

    let trainValidationCmd = metal.queue.makeCommandBuffer()!
    let trainValidationEnc = trainValidationCmd.makeComputeCommandEncoder()!
    metal.encodeForwardStepTrainFast(trainValidationEnc, stateIn: opInputShared, scratch: fastTrainValidationScratch,
                                     fc1WT: weights.fc1WT, fc1B: weights.fc1B, fc2WT: weights.fc2WT)
    metal.encodeBackwardStepFast(trainValidationEnc, dOutput: dOutputShared, scratch: fastTrainValidationScratch,
                                 fc1WT: weights.fc1WT, fc2WT: weights.fc2WT)
    trainValidationEnc.endEncoding()
    commitAndWait(trainValidationCmd, label: "fast train validation")
    guard compare("fast_output", metal.readBuffer(fastTrainValidationScratch.output, count: BATCH * STATE), loadOpRef("output")) else {
        fatalError("Fast training forward path mismatched the reference output")
    }
    guard compare("fast_d_fc2_w", metal.readBuffer(fastTrainValidationScratch.dFC2W, count: C * HIDDEN), loadOpRef("d_fc2_w"), tolerance: 5e-4) else {
        fatalError("Fast training d_fc2_w mismatched the reference output")
    }
    guard compare("fast_d_fc1_raw", metal.readBuffer(fastTrainValidationScratch.dFC1Raw, count: BATCH * HIDDEN * HW), loadOpRef("d_fc1_raw")) else {
        fatalError("Fast training d_fc1_raw mismatched the reference output")
    }
    guard compare("fast_d_fc1_b", metal.readBuffer(fastTrainValidationScratch.dFC1B, count: HIDDEN), loadOpRef("d_fc1_b"), tolerance: 5e-4) else {
        fatalError("Fast training d_fc1_b mismatched the reference output")
    }
    guard compare("fast_d_fc1_w", metal.readBuffer(fastTrainValidationScratch.dFC1W, count: HIDDEN * PERC), loadOpRef("d_fc1_w"), tolerance: 5e-4) else {
        fatalError("Fast training d_fc1_w mismatched the reference output")
    }
    guard compare("fast_d_state", metal.readBuffer(fastTrainValidationScratch.dState, count: BATCH * STATE), loadOpRef("d_state"), tolerance: 5e-4) else {
        fatalError("Fast training d_state mismatched the reference output")
    }

    let rolloutSummary = benchmark("forward_rollout_resident_\(options.rolloutSteps)",
                                   iterations: options.benchmarkIterations,
                                   warmup: options.warmupIterations,
                                   workItemsPerIteration: options.rolloutSteps) {
        let cmd = metal.queue.makeCommandBuffer()!
        let seedBlit = cmd.makeBlitCommandEncoder()!
        seedBlit.copy(from: rolloutSeedPrivate, sourceOffset: 0, to: rolloutStateA, destinationOffset: 0, size: rolloutSeedPrivate.length)
        seedBlit.endEncoding()
        let enc = cmd.makeComputeCommandEncoder()!
        _ = metal.encodeForwardRolloutFast(enc, steps: options.rolloutSteps,
                                           stateA: rolloutStateA, stateB: rolloutStateB, scratch: rolloutScratch,
                                           fc1WT: weights.fc1WT, fc1B: weights.fc1B, fc2WT: weights.fc2WT)
        enc.endEncoding()
        return cmd
    }
    printBenchmark(rolloutSummary)

    let fusedRolloutSummary = benchmark("forward_rollout_fused2_\(options.rolloutSteps)",
                                        iterations: options.benchmarkIterations,
                                        warmup: options.warmupIterations,
                                        workItemsPerIteration: options.rolloutSteps) {
        let cmd = metal.queue.makeCommandBuffer()!
        let seedBlit = cmd.makeBlitCommandEncoder()!
        seedBlit.copy(from: rolloutSeedPrivate, sourceOffset: 0, to: rolloutStateA, destinationOffset: 0, size: rolloutSeedPrivate.length)
        seedBlit.endEncoding()
        let enc = cmd.makeComputeCommandEncoder()!
        _ = metal.encodeForwardRolloutFused2(enc, steps: options.rolloutSteps,
                                             stateA: rolloutStateA, stateB: rolloutStateB, scratch: rolloutScratch,
                                             fc1WT: weights.fc1WT, fc1B: weights.fc1B, fc2WT: weights.fc2WT)
        enc.endEncoding()
        return cmd
    }
    printBenchmark(fusedRolloutSummary)

    let trainLikeSummary = benchmark("forward_backward_1",
                                     iterations: options.benchmarkIterations,
                                     warmup: options.warmupIterations,
                                     workItemsPerIteration: 1) {
        let cmd = metal.queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        metal.encodeForwardStepTrainFast(enc, stateIn: opInputPrivate, scratch: fastTrainBenchmarkScratch,
                                         fc1WT: weights.fc1WT, fc1B: weights.fc1B, fc2WT: weights.fc2WT)
        metal.encodeBackwardStepFast(enc, dOutput: dOutputPrivate, scratch: fastTrainBenchmarkScratch,
                                     fc1WT: weights.fc1WT, fc2WT: weights.fc2WT)
        enc.endEncoding()
        return cmd
    }
    printBenchmark(trainLikeSummary)

    let forwardOnlySummary = benchmark("forward_only_1",
                                       iterations: options.benchmarkIterations,
                                       warmup: options.warmupIterations,
                                       workItemsPerIteration: 1) {
        let cmd = metal.queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        metal.encodeForwardStepTrainFast(enc, stateIn: opInputPrivate, scratch: fastTrainBenchmarkScratch,
                                         fc1WT: weights.fc1WT, fc1B: weights.fc1B, fc2WT: weights.fc2WT)
        enc.endEncoding()
        return cmd
    }
    printBenchmark(forwardOnlySummary)

    let backwardPrepCmd = metal.queue.makeCommandBuffer()!
    let backwardPrepEnc = backwardPrepCmd.makeComputeCommandEncoder()!
    metal.encodeForwardStepTrainFast(backwardPrepEnc, stateIn: opInputPrivate, scratch: fastTrainBenchmarkScratch,
                                     fc1WT: weights.fc1WT, fc1B: weights.fc1B, fc2WT: weights.fc2WT)
    backwardPrepEnc.endEncoding()
    commitAndWait(backwardPrepCmd, label: "fast backward prep")

    let backwardCoreSummary = benchmark("backward_core_1",
                                        iterations: options.benchmarkIterations,
                                        warmup: options.warmupIterations,
                                        workItemsPerIteration: 1) {
        let cmd = metal.queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        metal.encodeBackwardCoreFast(enc, dOutput: dOutputPrivate, scratch: fastTrainBenchmarkScratch,
                                     fc1WT: weights.fc1WT, fc2WT: weights.fc2WT)
        enc.endEncoding()
        return cmd
    }
    printBenchmark(backwardCoreSummary)

    let weightPrepCmd = metal.queue.makeCommandBuffer()!
    let weightPrepEnc = weightPrepCmd.makeComputeCommandEncoder()!
    metal.encodeForwardStepTrainFast(weightPrepEnc, stateIn: opInputPrivate, scratch: fastTrainBenchmarkScratch,
                                     fc1WT: weights.fc1WT, fc1B: weights.fc1B, fc2WT: weights.fc2WT)
    metal.encodeBackwardWeightGradPrepFast(weightPrepEnc, dOutput: dOutputPrivate, scratch: fastTrainBenchmarkScratch,
                                           fc2WT: weights.fc2WT)
    weightPrepEnc.endEncoding()
    commitAndWait(weightPrepCmd, label: "fast weight-grad prep")

    let weightGradSummary = benchmark("weight_grads_1",
                                      iterations: options.benchmarkIterations,
                                      warmup: options.warmupIterations,
                                      workItemsPerIteration: 1) {
        let cmd = metal.queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        metal.encodeBackwardWeightGradFast(enc, scratch: fastTrainBenchmarkScratch)
        enc.endEncoding()
        return cmd
    }
    printBenchmark(weightGradSummary)
}

// MARK: - Main
let options = parseOptions()
log("Starting...")
let metal = NCAMetal()
log("Metal initialized")
print("Device: \(metal.device.name)")
let weights = loadWeights(metal)

var exitCode: Int32 = 0

switch options.mode {
case .accuracy:
    if !(verifySingleStep(metal, weights) && verifyRollout(metal, weights, steps: options.rolloutSteps)) {
        exitCode = 1
    }
case .benchmark:
    runBenchmarks(metal, weights, options: options)
case .all:
    let accuracyOK = verifySingleStep(metal, weights) && verifyRollout(metal, weights, steps: options.rolloutSteps)
    if accuracyOK {
        runBenchmarks(metal, weights, options: options)
    } else {
        exitCode = 1
    }
}

exit(exitCode)
