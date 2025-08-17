import Foundation
import CoreML

func loadMLModel(at path: String) throws -> MLModel {
    let url = URL(fileURLWithPath: path)
    let compiledURL = try MLModel.compileModel(at: url)
    let config = MLModelConfiguration()
    config.computeUnits = .all
    return try MLModel(contentsOf: compiledURL, configuration: config)
}

func runOnce(model: MLModel) throws {
    // Match export shapes
    let chunkLength = 256
    let featureDim = 80
    let dModel = 256

    // audio_chunk: (1, T, F)
    let audioChunk = try MLMultiArray(shape: [1, NSNumber(value: chunkLength), NSNumber(value: featureDim)], dataType: .float32)
    // token_in: (1, 1)
    let tokenIn = try MLMultiArray(shape: [1, 1], dataType: .int32)
    tokenIn[0] = 0
    // predictor_hidden_in: (1, 1, dModel)
    let hiddenIn = try MLMultiArray(shape: [1, 1, NSNumber(value: dModel)], dataType: .float32)

    let inputDict: [String: Any] = [
        "audio_chunk": audioChunk,
        "token_in": tokenIn,
        "predictor_hidden_in": hiddenIn
    ]

    let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputDict))
    guard let logits = out.featureValue(for: "logits_time")?.multiArrayValue,
          let hiddenOut = out.featureValue(for: "predictor_hidden_out")?.multiArrayValue else {
        print("Missing outputs from model")
        return
    }
    print("logits_time shape: \(logits.shape)")
    print("predictor_hidden_out shape: \(hiddenOut.shape)")
}

let args = CommandLine.arguments
let mlpackagePath: String
let compiledPath: String
if args.count > 1 {
    mlpackagePath = args[1]
    compiledPath = (args.count > 2) ? args[2] : "../../MambaASR.mlmodelc"
} else {
    // default relative to repo root
    mlpackagePath = "../../MambaASR.mlpackage"
    compiledPath = "../../MambaASR.mlmodelc"
}

do {
    // prefer compiled path if available
    let compiledURL = URL(fileURLWithPath: compiledPath)
    let model: MLModel
    if FileManager.default.fileExists(atPath: compiledURL.path) {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        model = try MLModel(contentsOf: compiledURL, configuration: config)
    } else {
        model = try loadMLModel(at: mlpackagePath)
    }
    try runOnce(model: model)
    print("Inference OK")
} catch {
    fputs("Error: \(error)\n", stderr)
    exit(1)
}
