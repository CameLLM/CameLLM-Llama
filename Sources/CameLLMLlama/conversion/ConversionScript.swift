//
//  ConversionScript.swift
//
//
//  Created by Alex Rozanski on 22/04/2023.
//

import Foundation
import CameLLMPluginHarness

enum ConversionScript {
  case genericConvertGgml
  case convertPyTorchToGgml
  case convertLoraToGgml

  var name: String {
    switch self {
    case .genericConvertGgml: return "convert"
    case .convertPyTorchToGgml: return "convert-pth-to-ggml"
    case .convertLoraToGgml: return "convert-lora-to-ggml"
    }
  }

  var pythonScript: PythonScript? {
    return Bundle.module.url(forResource: name, withExtension: "py").map {
      PythonScript(url: $0, pythonDependencies: pythonDependencies, dependentScripts: dependentScripts)
    }
  }

  var pythonDependencies: [String] {
    switch self {
    case .genericConvertGgml:
      return ["numpy", "sentencepiece"]
    case .convertPyTorchToGgml:
      return ["numpy", "sentencepiece", "torch"]
    case .convertLoraToGgml:
      return ["numpy", "sentencepiece", "torch"]
    }
  }

  var dependentScripts: [PythonScript] {
    switch self {
    case .genericConvertGgml:
      return []
    case .convertPyTorchToGgml:
      return [ConversionScript.genericConvertGgml.pythonScript].compactMap { $0 }
    case .convertLoraToGgml:
      return [ConversionScript.genericConvertGgml.pythonScript].compactMap { $0 }
    }
  }
}

