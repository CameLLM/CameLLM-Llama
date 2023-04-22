//
//  ModelConverter+Llama.swift
//
//
//  Created by Alex Rozanski on 22/04/2023.
//

import Foundation
import CameLLM
import CameLLMPluginHarness
import CameLLMLlamaObjCxx

public extension ModelConverter {
  static let llamaFamily = LlamaFamilyModelConverter()
}

public class LlamaFamilyModelConverter: ModelTypeScopedConversion {
  fileprivate init() {}

  // MARK: - Validation

  public func validateConversionData(_ data: ConvertPyTorchToGgmlConversionData, returning outRequiredFiles: inout [ModelConversionFile]?) -> Result<ValidatedModelConversionData<ConvertPyTorchToGgmlConversionData>, ConvertPyTorchToGgmlConversionData.ValidationError> {
    return ConvertPyTorchToGgmlConversion.validate(data, returning: &outRequiredFiles)
  }

  // MARK: - Conversion

  public func canRunConversion() async throws -> Bool {
    return try await ModelConversionUtils.shared.checkConversionEnvironment(input: (), connectors: makeEmptyConnectors()).isSuccess
  }

  public func makeConversionPipeline() -> ModelConversionPipeline<
    ConvertPyTorchToGgmlConversionStep,
    ConvertPyTorchToGgmlConversionPipelineInput,
    ConvertPyTorchToGgmlConversionResult
  > {
    return ConvertPyTorchToGgmlConversion().makeConversionPipeline()
  }

  // MARK: - Quantization

  public func quantizeModel(from sourceFileURL: URL, to destinationFileURL: URL) async throws {
    return try await withCheckedThrowingContinuation { continuation in
      do {
        try _LlamaModelUtils.quantizeModel(withSourceFileURL: sourceFileURL, destFileURL: destinationFileURL, fileType: .mostlyQ4_0)
        continuation.resume(returning: ())
      } catch {
        continuation.resume(throwing: error)
      }
    }
  }
}
