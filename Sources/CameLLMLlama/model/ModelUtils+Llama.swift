//
//  ModelUtils+Llama.swift
//  llama
//
//  Created by Alex Rozanski on 05/04/2023.
//

import Foundation
import CameLLM
import CameLLMPluginHarness
import CameLLMLlamaObjCxx
import CameLLMObjCxx

public extension ModelUtils {
  static let llamaFamily = LlamaFamilyModelUtils()
}

public final class LlamaFamilyModelCard: ModelCard {
  public let modelType: ModelType

  init(modelType: ModelType) {
    self.modelType = modelType

    let parameters: ParameterSize
    switch modelType {
    case .unknown:
      parameters = .billions(Decimal(0))
    case .size7B:
      parameters = .billions(Decimal(7))
    case .size13B:
      parameters = .billions(Decimal(13))
    case .size30B:
      parameters = .billions(Decimal(30))
    case .size65B:
      parameters = .billions(Decimal(65))
    }

    super.init(parameters: parameters)
  }
}

public class LlamaFamilyModelUtils: ModelTypeScopedUtils {
  public func getModelCard(forFileAt fileURL: URL) throws -> LlamaFamilyModelCard? {
    var modelType: _LlamaModelType = .typeUnknown
    try _LlamaModelUtils.loadModelTypeForFile(at: fileURL, outModelType: &modelType)
    return ModelType.fromObjCxxModelType(modelType).map { LlamaFamilyModelCard.init(modelType: $0) }
  }

  fileprivate init() {}

  public func validateModel(at fileURL: URL) throws {
    do {
      guard let _ = try getModelCard(forFileAt: fileURL) else {
        throw NSError(domain: CameLLMError.Domain, code: CameLLMError.Code.failedToValidateModel.rawValue)
      }
    } catch {
      let error = error as NSError

      // Since we get the model type by loading the file, failures should be `failedToLoadModel`.
      guard error.domain == CameLLMError.Domain, error.code == CameLLMError.Code.failedToLoadModel.rawValue else {
        throw error
      }

      // Retag this as `failedToValidateModel` for a more consistent API
      throw NSError(domain: CameLLMError.Domain, code: CameLLMError.Code.failedToValidateModel.rawValue, userInfo: error.userInfo)
    }
  }
}

fileprivate extension ModelType {
  static func fromObjCxxModelType(_ type: _LlamaModelType) -> ModelType? {
    switch type {
    case .typeUnknown:
      return nil
    case .type7B:
      return .size7B
    case .type13B:
      return .size13B
    case .type30B:
      return .size30B
    case .type65B:
      return .size65B
    default:
      return nil
    }
  }
}
