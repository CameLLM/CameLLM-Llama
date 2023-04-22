//
//  LlamaPredictOperation.h
//  llama
//
//  Created by Alex Rozanski on 13/03/2023.
//

#import <Foundation/NSOperation.h>

@class _CameLLMPredictionEvent;
@class LlamaContext;
@class _CameLLMSessionContext;

NS_ASSUME_NONNULL_BEGIN

typedef void (^LlamaPredictOperationEventHandler)(_CameLLMPredictionEvent *event);

__attribute__((visibility("default")))
@interface LlamaPredictOperation : NSOperation

@property (nonatomic, readonly, copy) NSString *identifier;

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

- (instancetype)initWithIdentifier:(NSString *)identifier
                           context:(LlamaContext *)context
                            prompt:(NSString *)prompt
                      eventHandler:(LlamaPredictOperationEventHandler)eventHandler;

@end

NS_ASSUME_NONNULL_END
