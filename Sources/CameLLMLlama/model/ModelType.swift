//
//  ModelType.swift
//  llama
//
//  Created by Alex Rozanski on 02/04/2023.
//

import CameLLM
import Foundation

public enum ModelType {
  case unknown
  case size7B
  case size13B
  case size30B
  case size65B
}

public extension ModelType {
  static func from(parameters: ParameterSize) -> ModelType? {
    if parameters == .billions(Decimal(7)) {
      return .size7B
    }

    if parameters == .billions(Decimal(13)) {
      return .size13B
    }

    if parameters == .billions(Decimal(30)) {
      return .size30B
    }

    if parameters == .billions(Decimal(65)) {
      return .size65B
    }

    return .unknown
  }
}
