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
  public func makeSession(
    with modelURL: URL,
    config: SessionConfig
  ) -> any Session<LlamaSessionState, LlamaPredictionState> {
    return LlamaSession(modelURL: modelURL, paramsBuilder: config)
  }
}
