//
//  LlamaSetupOperation.h
//  llamaObjCxx
//
//  Created by Alex Rozanski on 23/03/2023.
//

#import <Foundation/Foundation.h>

@class _LlamaSessionParams;
@class _CameLLMSetupEvent<ContextType>;
@class LlamaContext;
@class LlamaSetupOperation;

NS_ASSUME_NONNULL_BEGIN

typedef void (^LlamaSetupOperationEventHandler)(_CameLLMSetupEvent<LlamaContext *> *event);

__attribute__((visibility("default")))
@interface LlamaSetupOperation : NSOperation

@property (nonatomic, readonly, copy) LlamaSetupOperationEventHandler eventHandler;

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

- (instancetype)initWithParams:(_LlamaSessionParams *)params eventHandler:(LlamaSetupOperationEventHandler)eventHandler;

@end

NS_ASSUME_NONNULL_END
