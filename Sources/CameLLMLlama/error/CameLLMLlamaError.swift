//
//  CameLLMLlamaError.swift
//  llama
//
//  Created by Alex Rozanski on 05/04/2023.
//

import Foundation
import CameLLMLlamaObjCxx

public struct CameLLMLlamaError {
  public typealias Code = _CameLLMLlamaErrorCode
  public static let Domain = _CameLLMLlamaErrorDomain

  private init() {}
}
