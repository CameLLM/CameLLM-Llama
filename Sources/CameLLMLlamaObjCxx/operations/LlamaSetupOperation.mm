//
//  LlamaSetupOperation.m
//  llamaObjCxx
//
//  Created by Alex Rozanski on 23/03/2023.
//

@import CameLLMObjCxx;
@import CameLLMPluginHarnessObjCxx;
@import CameLLMCommonObjCxx;

#import "LlamaSetupOperation.h"

#import "LlamaError.h"
#import "LlamaContext.h"
#import "LlamaContext+Internal.hh"
#import "LlamaOperationUtils.hh"
#import "LlamaSessionParams.h"

#include "ggml.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#endif

@interface LlamaSetupOperation () {
  _LlamaSessionParams *_params;
  LlamaSetupOperationEventHandler _eventHandler;
}

@end

@implementation LlamaSetupOperation

- (instancetype)initWithParams:(_LlamaSessionParams *)params eventHandler:(LlamaSetupOperationEventHandler)eventHandler
{
  if ((self = [super init])) {
    _params = params;
    _eventHandler = [eventHandler copy];
  }

  return self;
}

- (void)main
{
  LlamaContext *context = nil;
  NSError *setUpError = nil;

  if (![self _setUpReturningContext:&context error:&setUpError]) {
    if (_eventHandler) {
      _eventHandler([_CameLLMSetupEvent failedWithError:setUpError]);
    }
  } else {
    _eventHandler([_CameLLMSetupEvent succeededWithContext:context]);
  }
}

- (BOOL)_setUpReturningContext:(LlamaContext **)outContext error:(NSError **)outError
{
  if (_params.contextSize > 2048) {
    NSLog(@"warning: model does not support context sizes greater than 2048 tokens (%d specified);"
          "expect poor results", _params.contextSize);
  }

  llama_context * ctx;

  // load the model
  {
    auto lparams = llama_context_default_params();

    lparams.n_ctx        = _params.contextSize;
    lparams.n_gpu_layers = _params.numberOfThreads;
    lparams.seed         = _params.seed;
    lparams.f16_kv       = _params.useF16Memory;
    lparams.use_mmap     = _params.useMmap;
    lparams.use_mlock    = _params.useMlock;
    const char *modelPath = [_params.modelPath cStringUsingEncoding:NSUTF8StringEncoding];
    ctx = llama_init_from_file(modelPath, lparams, outError);
    if (ctx == NULL) {
      return NO;
    }
  }

  if (_params.loraAdapter.length > 0) {
    std::string lora_adapter([_params.loraAdapter cStringUsingEncoding:NSUTF8StringEncoding]);
    std::string lora_base(_params.loraBase.length > 0 ? [_params.loraBase cStringUsingEncoding:NSUTF8StringEncoding] : "");
    int err = llama_apply_lora_from_file(ctx,
                                          lora_adapter.c_str(),
                                          lora_base.empty() ? NULL : lora_base.c_str(),
                                          _params.numberOfThreads);
    if (err != 0) {
      if (outError) {
        *outError = makeFailedToPredictErrorWithUnderlyingError(makeError(_CameLLMLlamaErrorDomain, _CameLLMLlamaErrorCodeFailedToApplyLoraAdapter, [NSString stringWithFormat:@"failed to apply lora adapter"], nil));
      }
      return NO;
    }
}

  LlamaContext *context = [[LlamaContext alloc] initWithParams:_params context:ctx];
  NSString *initialPrompt = @"";
  if (context.params.initialPrompt) {
    initialPrompt = context.params.initialPrompt;
  }

  std::string prompt([initialPrompt cStringUsingEncoding:NSUTF8StringEncoding]);

  // Add a space in front of the first character to match OG llama tokenizer behavior
  prompt.insert(0, 1, ' ');

  // tokenize the initial prompt
  if (![LlamaOperationUtils tokenizeString:prompt with:context into:context.runState->embd_inp addBeginningOfSequence:true outError:outError]) {
    return NO;
  }

  // Initialize the run state.
  const int n_ctx = llama_n_ctx(context.ctx);

  // Remaining setup.
  if ((int)context.runState->embd_inp.size() > n_ctx - 4) {
    if (outError) {
      *outError = makeFailedToPredictErrorWithUnderlyingError(makeError(_CameLLMLlamaErrorDomain, _CameLLMLlamaErrorCodePromptIsTooLong, [NSString stringWithFormat:@"prompt is too long (%d tokens, max %d)\n", (int)context.runState->embd_inp.size(), n_ctx - 4], nil));
    }
    return NO;
  }

  auto runState = context.runState;

  // number of tokens to keep when resetting context
  if (context.params.numberOfTokensToKeepFromInitialPrompt < 0 || context.params.numberOfTokensToKeepFromInitialPrompt > (int)context.runState->embd_inp.size() || _params.isInstructional) {
    context.params.numberOfTokensToKeepFromInitialPrompt = (int)context.runState->embd_inp.size();
  }

  // TODO: replace with ring-buffer
  context.runState->last_n_tokens.resize(n_ctx);
  std::fill(context.runState->last_n_tokens.begin(), context.runState->last_n_tokens.end(), 0);

  runState->n_past = 0;
  runState->n_remain = context.params.numberOfTokens;
  runState->n_consumed = 0;

  runState->is_antiprompt = false;

  if (outContext != NULL) {
    *outContext = context;
  }

  return YES;
}

@end
