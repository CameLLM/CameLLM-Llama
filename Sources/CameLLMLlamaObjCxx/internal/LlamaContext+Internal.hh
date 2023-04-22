//
//  LlamaContext+Internal.h
//  
//
//  Created by Alex Rozanski on 22/04/2023.
//

#import "LlamaContext.h"
#include <vector>

#import "llama.hh"
#import "LlamaRunState.h"

@class _LlamaSessionParams;
@class LlamaContext;

NS_ASSUME_NONNULL_BEGIN

// Expose these properties in an extension so that LlamaContext can be
// imported by Swift code that doesn't need access to the C++ members.
@interface LlamaContext ()

@property (nonatomic, readonly) _LlamaSessionParams *params;

// Context from Llama internal implementation.
@property (nonatomic, readonly, assign, nullable) llama_context *ctx;

// Run state shared between run invocations.
@property (nonatomic, readonly, assign, nullable) llama_swift_run_state *runState;

- (instancetype)initWithParams:(_LlamaSessionParams *)params context:(llama_context *)ctx;

@end

NS_ASSUME_NONNULL_END
