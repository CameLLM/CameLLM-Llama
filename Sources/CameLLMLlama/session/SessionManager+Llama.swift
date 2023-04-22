//
//  SessionManager+Llama.swift
//
//
//  Created by Alex Rozanski on 22/04/2023.
//

import Foundation
import CameLLM
import CameLLMCommon
import CameLLMPluginHarness

public extension SessionManager {
  static let llamaFamily = LlamaFamilySessionManager()
}

public class LlamaFamilySessionManager {
  public func makeLlamaSession(
    with modelURL: URL,
    config: LlamaSessionConfig
  ) -> any Session<LlamaSessionState, LlamaPredictionState> {
    return LlamaSession(modelURL: modelURL, paramsBuilder: config)
  }

  public func makeAlpacaSession(
    with modelURL: URL,
    config: AlpacaSessionConfig
  ) -> any Session<LlamaSessionState, LlamaPredictionState> {
    return LlamaSession(modelURL: modelURL, paramsBuilder: config)
  }

  public func makeGPT4AllSession(
    with modelURL: URL,
    config: GPT4AllSessionConfig
  ) -> any Session<LlamaSessionState, LlamaPredictionState> {
    return LlamaSession(modelURL: modelURL, paramsBuilder: config)
  }
}
