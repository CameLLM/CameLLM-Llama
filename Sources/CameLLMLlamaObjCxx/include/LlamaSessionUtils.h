//
//  LlamaSessionUtils.h
//
//
//  Created by Alex Rozanski on 23/04/2023.
//

#import <Foundation/Foundation.h>

@class _SessionContext;
@class LlamaContext;

NS_ASSUME_NONNULL_BEGIN

@class _CameLLMSessionContext;

/* WARNING: These methods should only be run on the session operation queue to ensure correct access of LlamaContext. */
__attribute__((visibility("default")))
@interface LlamaSessionUtils : NSObject

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

+ (nullable _SessionContext *)currentSessionContextWithLlamaContext:(nullable LlamaContext *)context;

@end

NS_ASSUME_NONNULL_END
