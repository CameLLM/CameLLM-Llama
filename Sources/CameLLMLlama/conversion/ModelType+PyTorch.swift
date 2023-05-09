//
//  PyTorch.swift
//  
//
//  Created by Alex Rozanski on 09/05/2023.
//

import Foundation

public extension ModelType {
  var numPyTorchModelParts: Int {
    switch self {
    case .unknown:
      return 0
    case .size7B:
      return 1
    case .size13B:
      return 2
    case .size30B:
      return 4
    case .size65B:
      return 8
    }
  }
}
