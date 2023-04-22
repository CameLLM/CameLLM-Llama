//
//  LlamaErrorUtils.m
//
//
//  Created by Alex Rozanski on 23/04/2023.
//

@import CameLLMObjCxx;
@import CameLLMPluginHarnessObjCxx;

#import "LlamaErrorUtils.h"

#import "LlamaError.h"

NSError *makeFailedToQuantizeErrorWithUnderlyingError(NSError *__nullable underlyingError)
{
  return makeError(_CameLLMLlamaErrorDomain, _CameLLMLlamaErrorCodeFailedToQuantize, @"Failed to quantize model", underlyingError);
}
