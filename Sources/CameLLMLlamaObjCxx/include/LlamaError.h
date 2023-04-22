//
//  LlamaError.h
//  llama
//
//  Created by Alex Rozanski on 14/03/2023.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

extern NSString *const _CameLLMLlamaErrorDomain;

typedef NS_ENUM(NSInteger, _CameLLMLlamaErrorCode) {
  _CameLLMLlamaErrorCodeUnknown = -1,

  _CameLLMLlamaErrorCodeFailedToQuantize = -1001,
  _CameLLMLlamaErrorCodePromptIsTooLong = -1002,
  _CameLLMLlamaErrorCodeFailedToApplyLoraAdapter = -1003,


};

NS_ASSUME_NONNULL_END
