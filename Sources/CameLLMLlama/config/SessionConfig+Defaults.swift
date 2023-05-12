//
//  SessionConfig+Defaults.swift
//  
//
//  Created by Alex Rozanski on 12/05/2023.
//

import Foundation

public extension SessionConfig {
  static var alpaca: SessionConfig {
    return configurableAlpaca().build()
  }

  // Based on values in https://github.com/ggerganov/llama.cpp/blob/107980d/examples/alpaca.sh
  static func configurableAlpaca() -> SessionConfigBuilder {
    return SessionConfigBuilder(defaults: defaultSessionConfig)
      .withMode(.instructional)
      .withNumTokens(512)
      .withHyperparameters { hyperparameters in
        hyperparameters
          .withContextSize(2048)
          .withBatchSize(256)
          .withTopK(10000)
          .withTemperature(0.2)
          .withRepeatPenalty(1)
      }
      .withInitialPrompt("Below is an instruction that describes a task. Write a response that appropriately completes the request.")
      .withPromptPrefix("\n\n### Instruction:\n\n")
      .withPromptSuffix("\n\n### Response:\n\n")
      .withAntiprompt("### Instruction:\n\n")
  }

  static var llama: SessionConfig {
    return configurableAlpaca().build()
  }

  static func configurableLlama() -> SessionConfigBuilder {
    return SessionConfigBuilder(defaults: defaultSessionConfig)
      .withMode(.regular)
      .withNumTokens(128)
  }

  static var gpt4All: SessionConfig {
    return configurableAlpaca().build()
  }

  static func configurableGPT4All() -> SessionConfigBuilder {
    return SessionConfigBuilder(defaults: defaultSessionConfig)
      .withMode(.instructional)
      .withNumTokens(128)
      .withHyperparameters { hyperparameters in
        hyperparameters
          .withContextSize(2048)
          .withBatchSize(8)
          .withLastNTokensToPenalize(64)
          .withTopK(40)
          .withTopP(0.95)
          .withTemperature(0.1)
          .withRepeatPenalty(1.3)
      }
      .withInitialPrompt("Below is an instruction that describes a task. Write a response that appropriately completes the request.")
      .withPromptPrefix("\n\n### Instruction:\n\n")
      .withPromptSuffix("\n\n### Response:\n\n")
      .withAntiprompt("### Instruction:\n\n")
  }
}
