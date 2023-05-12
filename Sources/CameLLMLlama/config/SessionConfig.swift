//
//  SessionConfig.swift
//  llama
//
//  Created by Alex Rozanski on 29/03/2023.
//

import Foundation
import CameLLMLlamaObjCxx

public struct Hyperparameters {
  // The number of tokens to keep as context
  public fileprivate(set) var contextSize: UInt
  public fileprivate(set) var batchSize: UInt
  public fileprivate(set) var lastNTokensToPenalize: UInt
  public fileprivate(set) var topK: UInt
  // Should be between 0 and 1
  public fileprivate(set) var topP: Double
  public fileprivate(set) var temperature: Double
  public fileprivate(set) var repeatPenalty: Double

  public init(
    contextSize: UInt,
    batchSize: UInt,
    lastNTokensToPenalize: UInt,
    topK: UInt,
    topP: Double,
    temperature: Double,
    repeatPenalty: Double
  ) {
    self.contextSize = contextSize
    self.batchSize = batchSize
    self.lastNTokensToPenalize = lastNTokensToPenalize
    self.topK = topK
    self.topP = topP
    self.temperature = temperature
    self.repeatPenalty = repeatPenalty
  }
}

// MARK: -

public final class SessionConfig: ObjCxxParamsBuilder {
  public enum Mode {
    case regular
    case instructional

    func toObjCxxMode() -> _LlamaSessionMode {
      switch self {
      case .regular: return .regular
      case .instructional: return .instructional
      }
    }
  }

  public private(set) var mode: Mode

  // Seed for generation
  public private(set) var seed: Int32?

  // Number of threads to run prediction on.
  public private(set) var numThreads: UInt

  // Number of tokens to predict for each run.
  public private(set) var numTokens: UInt

  // Whether to load and keep the entire model in memory.
  public private(set) var keepModelInMemory: Bool

  // Model configuration
  public private(set) var hyperparameters: Hyperparameters

  public private(set) var initialPrompt: String?
  public private(set) var promptPrefix: String?
  public private(set) var promptSuffix: String?
  public private(set) var antiprompt: String?

  init(
    mode: Mode,
    seed: Int32? = nil,
    numThreads: UInt,
    numTokens: UInt,
    keepModelInMemory: Bool,
    hyperparameters: Hyperparameters,
    initialPrompt: String?,
    promptPrefix: String?,
    promptSuffix: String?,
    antiprompt: String?
  ) {
    self.mode = mode
    self.seed = seed
    self.numThreads = numThreads
    self.numTokens = numTokens
    self.keepModelInMemory = keepModelInMemory
    self.hyperparameters = hyperparameters
    self.initialPrompt = initialPrompt
    self.promptPrefix = promptPrefix
    self.promptSuffix = promptSuffix
    self.antiprompt = antiprompt
  }

  func build(for modelURL: URL) -> _LlamaSessionParams {
    let params = _LlamaSessionParams.defaultParams(withModelPath: modelURL.path, mode: mode.toObjCxxMode())
    params.numberOfThreads = Int32(numThreads)
    params.numberOfTokens = Int32(numTokens)

    if let seed = seed { params.seed = seed }
    params.contextSize = Int32(hyperparameters.contextSize)
    params.batchSize = Int32(hyperparameters.batchSize)
    params.lastNTokensToPenalize = Int32(hyperparameters.lastNTokensToPenalize)
    params.topP = Float(hyperparameters.topP)
    params.topK = Int32(hyperparameters.topK)
    params.temp = Float(hyperparameters.temperature)
    params.repeatPenalty = Float(hyperparameters.repeatPenalty)

    params.initialPrompt = initialPrompt
    params.promptPrefix = promptPrefix
    params.promptSuffix = promptSuffix
    params.antiprompts = [antiprompt].compactMap { $0 }

    return params
  }
}

// MARK: - Config Builders

public class HyperparametersBuilder {
  public private(set) var contextSize: UInt?
  public private(set) var batchSize: UInt?
  public private(set) var lastNTokensToPenalize: UInt?
  public private(set) var topK: UInt?
  public private(set) var topP: Double?
  public private(set) var temperature: Double?
  public private(set) var repeatPenalty: Double?

  private let defaults: Hyperparameters

  init(defaults: Hyperparameters) {
    self.defaults = defaults
  }

  public func withContextSize(_ contextSize: UInt?) -> Self {
    self.contextSize = contextSize
    return self
  }

  public func withBatchSize(_ batchSize: UInt?) -> Self {
    self.batchSize = batchSize
    return self
  }

  public func withLastNTokensToPenalize(_ lastNTokensToPenalize: UInt?) -> Self {
    self.lastNTokensToPenalize = lastNTokensToPenalize
    return self
  }

  public func withTopK(_ topK: UInt?) -> Self {
    self.topK = topK
    return self
  }

  public func withTopP(_ topP: Double?) -> Self {
    self.topP = topP
    return self
  }

  public func withTemperature(_ temperature: Double?) -> Self {
    self.temperature = temperature
    return self
  }

  public func withRepeatPenalty(_ repeatPenalty: Double?) -> Self {
    self.repeatPenalty = repeatPenalty
    return self
  }

  func build() -> Hyperparameters {
    return Hyperparameters(
      contextSize: contextSize ?? defaults.contextSize,
      batchSize: batchSize ?? defaults.batchSize,
      lastNTokensToPenalize: lastNTokensToPenalize ?? defaults.lastNTokensToPenalize,
      topK: topK ?? defaults.topK,
      topP: topP ?? defaults.topP,
      temperature: temperature ?? defaults.temperature,
      repeatPenalty: repeatPenalty ?? defaults.repeatPenalty
    )
  }
}

// MARK: -

public class SessionConfigBuilder {
  public private(set) var mode: SessionConfig.Mode?
  public private(set) var seed: Int32??
  public private(set) var numThreads: UInt?
  public private(set) var numTokens: UInt?
  public private(set) var keepModelInMemory: Bool?
  public private(set) var hyperparameters: HyperparametersBuilder
  public private(set) var initialPrompt: String??
  public private(set) var promptPrefix: String??
  public private(set) var promptSuffix: String??
  public private(set) var antiprompt: String??

  private let defaults: SessionConfig

  init(defaults: SessionConfig) {
    self.hyperparameters = HyperparametersBuilder(defaults: defaults.hyperparameters)
    self.defaults = defaults
  }

  public func withSeed(_ seed: Int32??) -> Self {
    self.seed = seed
    return self
  }

  public func withNumThreads(_ numThreads: UInt?) -> Self {
    self.numThreads = numThreads
    return self
  }

  public func withNumTokens(_ numTokens: UInt?) -> Self {
    self.numTokens = numTokens
    return self
  }

  public func withKeepModelInMemory(_ keepModelInMemory: Bool) -> Self {
    self.keepModelInMemory = keepModelInMemory
    return self
  }

  public func withHyperparameters(_ hyperParametersConfig: (HyperparametersBuilder) -> HyperparametersBuilder) -> Self {
    self.hyperparameters = hyperParametersConfig(hyperparameters)
    return self
  }

  public func withInitialPrompt(_ initialPrompt: String??) -> Self {
    self.initialPrompt = initialPrompt
    return self
  }

  public func withPromptPrefix(_ promptPrefix: String??) -> Self {
    self.promptPrefix = promptPrefix
    return self
  }

  public func withPromptSuffix(_ promptSuffix: String??) -> Self {
    self.promptSuffix = promptSuffix
    return self
  }

  public func withAntiprompt(_ antiprompt: String??) -> Self {
    self.antiprompt = antiprompt
    return self
  }

  // MARK: - Internal

  public func withMode(_ mode: SessionConfig.Mode?) -> Self {
    self.mode = mode
    return self
  }

  // MARK: - Build

  public func build() -> SessionConfig {
    return SessionConfig(
      mode: mode ?? .instructional,
      seed: seed ?? defaults.seed,
      numThreads: numThreads ?? defaults.numThreads,
      numTokens: numTokens ?? defaults.numTokens,
      keepModelInMemory: keepModelInMemory ?? defaults.keepModelInMemory,
      hyperparameters: hyperparameters.build(),
      initialPrompt: initialPrompt ?? defaults.initialPrompt,
      promptPrefix: promptPrefix ?? defaults.promptPrefix,
      promptSuffix: promptSuffix ?? defaults.promptSuffix,
      antiprompt: antiprompt ?? defaults.antiprompt
    )
  }
}

// MARK: - Params Builders

extension SessionConfig {
  static var defaultNumThreads: UInt {
    let processorCount = UInt(ProcessInfo().activeProcessorCount)
    // Account for main thread and worker thread. Specifying all active processors seems to introduce a lot of contention.
    let maxAvailableProcessors = processorCount - 2
    // Experimentally 6 also seems like a pretty good number.
    return min(maxAvailableProcessors, 6)
  }
}

let defaultSessionConfig = {
  let params = _LlamaSessionParams.defaultParams(withModelPath: "", mode: .regular)
  return SessionConfig(
    mode: .instructional,
    seed: params.seed == -1 ? nil : params.seed,
    numThreads: UInt(params.numberOfThreads),
    numTokens: UInt(params.numberOfTokens),
    keepModelInMemory: params.useMlock,
    hyperparameters: Hyperparameters(
      contextSize: UInt(params.contextSize),
      batchSize: UInt(params.batchSize),
      lastNTokensToPenalize: UInt(params.lastNTokensToPenalize),
      topK: UInt(params.topK),
      topP: Double(params.topP),
      temperature: Double(params.temp),
      repeatPenalty: Double(params.repeatPenalty)
    ),
    initialPrompt: nil,
    promptPrefix: nil,
    promptSuffix: nil,
    antiprompt: nil
  )
}()
