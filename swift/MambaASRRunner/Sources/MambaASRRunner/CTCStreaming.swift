import Foundation
import CoreML

struct CTCContract: Decodable {
    struct Model: Decodable {
        let dModel: Int
        let nBlocks: Int
        let stateDim: Int
        let vocabSize: Int
        let timeReduction: Int
    }

    struct Streaming: Decodable {
        let chunkFrames: Int
        let outputFrames: Int
        let boundaryPolicy: String
        let convContextFrames: Int
    }

    struct Mel: Decodable {
        let sampleRate: Int
        let nFFT: Int
        let winLength: Int
        let hopLength: Int
        let nMels: Int
        let fMin: Int
        let fMax: Int
        let center: Bool
        let power: Float
        let melScale: String
        let norm: String?
        let logScale: String
        let logFloor: Float
        let window: String
    }

    struct IOContract: Decodable {
        let audioInput: String
        let stateInput: String
        let logitsOutput: String
        let stateOutput: String
        let audioShape: [Int]
        let stateShape: [Int]
        let logitsShape: [Int]
    }

    let schemaVersion: Int
    let modelType: String
    let precision: String
    let model: Model
    let streaming: Streaming
    let mel: Mel
    let vocab: [String]
    let io: IOContract
}

enum CTCContractError: LocalizedError {
    case invalid(String)
    case audio(String)
    case model(String)

    var errorDescription: String? {
        switch self {
        case .invalid(let message): return "CTC contract mismatch: \(message)"
        case .audio(let message): return "CTC audio error: \(message)"
        case .model(let message): return "CTC model mismatch: \(message)"
        }
    }
}

struct CTCStreamingStats {
    let avgMs: Double
    let p50Ms: Double
    let p90Ms: Double
    let count: Int
}

func loadCTCContract(path: String, chunkOverride: Int?) throws -> CTCContract {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let contract = try JSONDecoder().decode(CTCContract.self, from: data)

    guard contract.schemaVersion == 1 else {
        throw CTCContractError.invalid(
            "unsupported schemaVersion \(contract.schemaVersion); expected 1"
        )
    }
    guard contract.modelType == "ctc29" else {
        throw CTCContractError.invalid(
            "modelType is \(contract.modelType); expected ctc29"
        )
    }
    guard contract.model.vocabSize == 29, contract.vocab.count == 29 else {
        throw CTCContractError.invalid("vocabulary must contain 29 symbols")
    }
    guard contract.streaming.chunkFrames > 0,
          contract.streaming.chunkFrames % contract.model.timeReduction == 0 else {
        throw CTCContractError.invalid(
            "chunkFrames must be positive and divisible by timeReduction"
        )
    }
    if let requested = chunkOverride,
       requested != contract.streaming.chunkFrames {
        throw CTCContractError.invalid(
            "--chunk \(requested) does not match contract chunkFrames "
            + "\(contract.streaming.chunkFrames)"
        )
    }
    let expectedAudioShape = [
        1,
        contract.streaming.chunkFrames + contract.streaming.convContextFrames,
        contract.mel.nMels,
    ]
    guard contract.io.audioShape == expectedAudioShape else {
        throw CTCContractError.invalid(
            "audioShape \(contract.io.audioShape) != \(expectedAudioShape)"
        )
    }
    let expectedStateShape = [
        contract.model.nBlocks,
        1,
        contract.model.dModel,
        contract.model.stateDim,
    ]
    guard contract.io.stateShape == expectedStateShape else {
        throw CTCContractError.invalid(
            "stateShape \(contract.io.stateShape) != \(expectedStateShape)"
        )
    }
    let expectedLogitsShape = [
        1,
        contract.streaming.outputFrames,
        contract.model.vocabSize,
    ]
    guard contract.io.logitsShape == expectedLogitsShape else {
        throw CTCContractError.invalid(
            "logitsShape \(contract.io.logitsShape) != \(expectedLogitsShape)"
        )
    }
    guard contract.mel.sampleRate == 16_000,
          contract.mel.nFFT == 512,
          contract.mel.winLength == 400,
          contract.mel.hopLength == 160,
          contract.mel.nMels == 80,
          contract.mel.fMin == 0,
          contract.mel.fMax == 8_000,
          contract.mel.center == false,
          contract.mel.power == 2.0,
          contract.mel.melScale == "htk",
          contract.mel.norm == nil,
          contract.mel.logScale == "natural",
          contract.mel.logFloor == 1e-6,
          contract.mel.window == "hann_periodic" else {
        throw CTCContractError.invalid(
            "mel parameters are not reproducible by this runner"
        )
    }
    guard contract.streaming.boundaryPolicy
            == "causal-conv-left-context-carry-mamba",
          contract.streaming.convContextFrames == 8 else {
        throw CTCContractError.invalid(
            "unsupported streaming boundary policy"
        )
    }
    return contract
}

private func multiArray(
    shape: [Int],
    dataType: MLMultiArrayDataType,
    values: [Float]? = nil
) throws -> MLMultiArray {
    let array = try MLMultiArray(
        shape: shape.map(NSNumber.init(value:)),
        dataType: dataType
    )
    if let values = values {
        guard values.count == array.count else {
            throw CTCContractError.model(
                "input value count \(values.count) != tensor count \(array.count)"
            )
        }
        for index in values.indices {
            array[index] = NSNumber(value: values[index])
        }
    } else {
        // MLMultiArray storage is not guaranteed to be zero-initialized.
        for index in 0..<array.count {
            array[index] = 0
        }
    }
    return array
}

private func tensorDataType(for precision: String) throws -> MLMultiArrayDataType {
    switch precision {
    case "fp32": return .float32
    case "fp16": return .float16
    default:
        throw CTCContractError.invalid("unsupported precision \(precision)")
    }
}

private func greedyCollapse(_ ids: [Int]) -> [Int] {
    var result: [Int] = []
    var previous: Int? = nil
    for id in ids {
        if id != 0 && id != previous {
            result.append(id)
        }
        previous = id
    }
    return result
}

private func writeLatencyCSV(path: String, latencies: [Double]) throws {
    let url = URL(fileURLWithPath: path)
    try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    var output = "chunk,latency_ms\n"
    for (index, latency) in latencies.enumerated() {
        output += "\(index),\(String(format: "%.3f", latency))\n"
    }
    try output.write(to: url, atomically: true, encoding: .utf8)
}

func runCTCStreaming(
    model: MLModel,
    contract: CTCContract,
    wavPath: String?,
    durationSeconds: Int,
    warmupCount: Int,
    latencyCSVPath: String?
) throws -> CTCStreamingStats {
    let mel = contract.mel
    let chunkFrames = contract.streaming.chunkFrames
    let chunkSamples = (chunkFrames - 1) * mel.hopLength + mel.nFFT
    let signal: [Float]
    if let wavPath = wavPath {
        guard let loaded = loadWavMono16k(path: wavPath) else {
            throw CTCContractError.audio("could not load \(wavPath) as 16 kHz WAV")
        }
        signal = loaded
    } else {
        let seconds = max(durationSeconds, 1)
        let requestedSamples = seconds * mel.sampleRate
        signal = generateSyntheticAudio(
            sampleRate: mel.sampleRate,
            samples: max(chunkSamples * 3, requestedSamples)
        )
    }

    let availableFrames = max(0, (signal.count - mel.nFFT) / mel.hopLength + 1)
    guard availableFrames > 0 else {
        throw CTCContractError.audio("audio is shorter than one analysis window")
    }
    let totalChunks = max(1, Int(ceil(Double(availableFrames) / Double(chunkFrames))))
    let dataType = try tensorDataType(for: contract.precision)
    var state = try multiArray(shape: contract.io.stateShape, dataType: dataType)
    var featureContext = [Float](
        repeating: 0,
        count: contract.streaming.convContextFrames * mel.nMels
    )
    var tokenIDs: [Int] = []
    var latencies: [Double] = []

    let warmAudio = try multiArray(
        shape: contract.io.audioShape,
        dataType: dataType,
        values: [Float](repeating: 0, count: contract.io.audioShape.reduce(1, *))
    )
    let warmInputs = [
        contract.io.audioInput: warmAudio,
        contract.io.stateInput: state,
    ]
    for _ in 0..<warmupCount {
        _ = try model.prediction(
            from: MLDictionaryFeatureProvider(dictionary: warmInputs)
        )
    }

    for chunkIndex in 0..<totalChunks {
        let firstFrame = chunkIndex * chunkFrames
        let framesThisChunk = min(chunkFrames, max(0, availableFrames - firstFrame))
        let firstSample = firstFrame * mel.hopLength
        let requiredSamples = (framesThisChunk - 1) * mel.hopLength + mel.nFFT
        var chunkSignal = Array(
            signal[firstSample..<min(signal.count, firstSample + requiredSamples)]
        )
        if framesThisChunk < chunkFrames {
            chunkSignal.append(
                contentsOf: [Float](
                    repeating: 0,
                    count: chunkSamples - chunkSignal.count
                )
            )
        }
        let newFeatures = computeLogMelSpectrogram(
            signal: chunkSignal,
            sampleRate: mel.sampleRate,
            nFFT: mel.nFFT,
            winLength: mel.winLength,
            hopLength: mel.hopLength,
            numMels: mel.nMels,
            numFrames: chunkFrames
        )
        guard newFeatures.count == chunkFrames * mel.nMels else {
            throw CTCContractError.audio(
                "feature extraction returned wrong new-chunk shape"
            )
        }
        if chunkIndex == 0,
           let minimum = newFeatures.min(),
           let maximum = newFeatures.max() {
            print(
                String(
                    format: "CTC features: min=%.6f max=%.6f first=%.6f",
                    minimum,
                    maximum,
                    newFeatures[0]
                )
            )
        }
        let modelFeatures = featureContext + newFeatures
        let audio = try multiArray(
            shape: contract.io.audioShape,
            dataType: dataType,
            values: modelFeatures
        )
        featureContext = Array(
            modelFeatures.suffix(
                contract.streaming.convContextFrames * mel.nMels
            )
        )
        let inputs = [
            contract.io.audioInput: audio,
            contract.io.stateInput: state,
        ]
        let start = CFAbsoluteTimeGetCurrent()
        let output = try model.prediction(
            from: MLDictionaryFeatureProvider(dictionary: inputs)
        )
        let elapsedMS = (CFAbsoluteTimeGetCurrent() - start) * 1_000
        latencies.append(elapsedMS)

        guard let logits = output.featureValue(
            for: contract.io.logitsOutput
        )?.multiArrayValue,
        let nextState = output.featureValue(
            for: contract.io.stateOutput
        )?.multiArrayValue else {
            throw CTCContractError.model(
                "missing outputs \(contract.io.logitsOutput), "
                + "\(contract.io.stateOutput)"
            )
        }
        guard logits.shape.map(\.intValue) == contract.io.logitsShape,
              nextState.shape.map(\.intValue) == contract.io.stateShape else {
            throw CTCContractError.model(
                "Core ML output shapes do not match contract"
            )
        }
        let validOutputFrames = Int(
            ceil(Double(framesThisChunk) / Double(contract.model.timeReduction))
        )
        for frame in 0..<min(validOutputFrames, contract.streaming.outputFrames) {
            var bestID = 0
            var bestValue = -Float.greatestFiniteMagnitude
            for id in 0..<contract.model.vocabSize {
                let value = logits[
                    [
                        NSNumber(value: 0),
                        NSNumber(value: frame),
                        NSNumber(value: id),
                    ]
                ].floatValue
                if value > bestValue {
                    bestValue = value
                    bestID = id
                }
            }
            tokenIDs.append(bestID)
        }
        state = nextState
        print(
            String(
                format: "[ctc chunk %d] latency=%.2f ms",
                chunkIndex,
                elapsedMS
            )
        )
    }

    let transcript = greedyCollapse(tokenIDs).map { contract.vocab[$0] }.joined()
    print("CTC greedy transcript: \(transcript)")
    let sorted = latencies.sorted()
    let average = sorted.reduce(0, +) / Double(sorted.count)
    let p50 = sorted[Int(Double(sorted.count - 1) * 0.5)]
    let p90 = sorted[Int(Double(sorted.count - 1) * 0.9)]
    print(
        String(
            format: "CTC latency: avg=%.2f ms p50=%.2f ms p90=%.2f ms (n=%d)",
            average,
            p50,
            p90,
            sorted.count
        )
    )
    if let path = latencyCSVPath, !path.isEmpty {
        try writeLatencyCSV(path: path, latencies: latencies)
        print("Latencies written to \(path)")
    }
    return CTCStreamingStats(
        avgMs: average,
        p50Ms: p50,
        p90Ms: p90,
        count: sorted.count
    )
}
