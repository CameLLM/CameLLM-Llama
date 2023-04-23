//
//  LlamaPredictOperation.m
//  llama
//
//  Created by Alex Rozanski on 13/03/2023.
//

@import CameLLMObjCxx;
@import CameLLMPluginHarnessObjCxx;
@import CameLLMCommonObjCxx;

#import "LlamaPredictOperation.h"

#import "LlamaContext.h"
#import "LlamaContext+Internal.hh"
#import "LlamaOperationUtils.hh"
#import "LlamaSessionParams.h"
#import "LlamaSessionUtils.h"

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

@interface LlamaPredictOperation () {
  LlamaContext *_context;
  NSString *_prompt;
  LlamaPredictOperationEventHandler _eventHandler;
}

@end

@implementation LlamaPredictOperation

- (instancetype)initWithIdentifier:(NSString *)identifier
                           context:(LlamaContext *)context
                            prompt:(NSString *)prompt
                      eventHandler:(LlamaPredictOperationEventHandler)eventHandler
{
  if ((self = [super init])) {
    _identifier = [identifier copy];
    _context = context;
    _prompt = [prompt copy];
    _eventHandler = [eventHandler copy];
  }

  return self;
}

- (void)main
{
  [self _postEvent:[_CameLLMPredictionEvent started]];

  if ([self _runPrediction]) {
    [self _postEvent:[_CameLLMPredictionEvent updatedSessionContext:[LlamaSessionUtils currentSessionContextWithLlamaContext:_context]]];
    [self _postEvent:self.isCancelled ? [_CameLLMPredictionEvent cancelled] : [_CameLLMPredictionEvent completed]];
  }
}

- (BOOL)_runPrediction
{
  const int n_ctx = llama_n_ctx(_context.ctx);

  BOOL needsToInjectPrompt = YES;

  // maps to ignore_noecho in llama.cpp
  // set to YES by default because of any initial prompts we inject.
  BOOL ignoreOutputtedTokens = YES;

  NSError *tokenizeError = nil;

  auto params = _context.params;
  auto runState = _context.runState;

  // prefix & suffix for instruct mode
  BOOL hasPrefix = params.promptPrefix != nil && params.promptPrefix.length > 0;
  std::vector<llama_token> inp_pfx;

  if (hasPrefix) {
    std::string prefix([params.promptPrefix cStringUsingEncoding:NSUTF8StringEncoding]);
    if (![LlamaOperationUtils tokenizeString:prefix with:_context into:inp_pfx addBeginningOfSequence:true outError:&tokenizeError]) {
      [self _postEvent:[_CameLLMPredictionEvent failedWithError:tokenizeError]];
      return NO;
    }
  }

  BOOL hasSuffix = params.promptSuffix != nil && params.promptSuffix.length > 0;
  std::vector<llama_token> inp_sfx;

  if (hasSuffix) {
    std::string suffix([params.promptSuffix cStringUsingEncoding:NSUTF8StringEncoding]);
    if (![LlamaOperationUtils tokenizeString:suffix with:_context into:inp_sfx addBeginningOfSequence:false outError:&tokenizeError]) {
      [self _postEvent:[_CameLLMPredictionEvent failedWithError:tokenizeError]];
      return NO;
    }
  }

  // determine newline token
  std::vector<llama_token> llama_token_newline;
  if (![LlamaOperationUtils tokenizeString:"\n" with:_context into:llama_token_newline addBeginningOfSequence:false outError:&tokenizeError]) {
    [self _postEvent:[_CameLLMPredictionEvent failedWithError:tokenizeError]];
    return NO;
  }

  // run in interactive mode always so run the loop until we are finished.
  while (runState->n_remain != 0) {
    if (self.isCancelled) {
      // even though we're cancelling this is still successful.
      return YES;
    }

    // predict
    if (runState->embd.size() > 0) {
      // infinite text generation via context swapping
      // if we run out of context:
      // - take the n_keep first tokens from the original prompt (via n_past)
      // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch
      if (runState->n_past + (int)runState->embd.size() > n_ctx) {
        const int n_left = runState->n_past - params.numberOfTokensToKeepFromInitialPrompt;

        runState->n_past = params.numberOfTokensToKeepFromInitialPrompt;

        // insert n_left/2 tokens at the start of embd from last_n_tokens
        runState->embd.insert(runState->embd.begin(), runState->last_n_tokens.begin() + n_ctx - n_left / 2 - runState->embd.size(), runState->last_n_tokens.end() - runState->embd.size());
      }

      if (llama_eval(_context.ctx, runState->embd.data(), (int)runState->embd.size(), runState->n_past, params.numberOfThreads)) {
        [self _postEvent:[_CameLLMPredictionEvent failedWithError:makeFailedToPredictErrorWithUnderlyingError(makeCameLLMError(_CameLLMErrorCodeGeneralInternalPredictionFailure, @"failed to eval"))]];
        return NO;
      }
    }

    runState->n_past += runState->embd.size();
    runState->embd.clear();

    if ((int)runState->embd_inp.size() <= runState->n_consumed && !needsToInjectPrompt) {
      // out of user input, sample next token
      const float top_k = params.topK;
      const float top_p = params.topP;
      const float temp = params.temp;
      const float repeat_penalty = params.repeatPenalty;

      llama_token id = 0;

      {
        auto logits = llama_get_logits(_context.ctx);

        id = llama_sample_top_p_top_k(_context.ctx,
                                      runState->last_n_tokens.data() + n_ctx - params.lastNTokensToPenalize,
                                      params.lastNTokensToPenalize, top_k, top_p, temp, repeat_penalty);

        runState->last_n_tokens.erase(runState->last_n_tokens.begin());
        runState->last_n_tokens.push_back(id);
      }

      // replace end of text token with newline token when in interactive mode
      if (id == llama_token_eos() && !params.isInstructional) {
        id = llama_token_newline.front();
        NSString *firstAntiprompt = params.antiprompts.firstObject;
        if (firstAntiprompt != nil) {
          // tokenize and inject first reverse prompt
          std::vector<llama_token> first_antiprompt;
          if (![LlamaOperationUtils tokenizeString:[firstAntiprompt cStringUsingEncoding:NSUTF8StringEncoding] with:_context into:first_antiprompt addBeginningOfSequence:false outError:&tokenizeError]) {
            [self _postEvent:[_CameLLMPredictionEvent failedWithError:tokenizeError]];
            return NO;
          }
          runState->embd_inp.insert(runState->embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
        }
      }

      // add it to the context
      runState->embd.push_back(id);

      // echo this to console
      ignoreOutputtedTokens = NO;

      // decrement remaining sampling budget
      --runState->n_remain;
    } else {
      // some user input remains from prompt or interaction, forward it to processing
      while ((int)runState->embd_inp.size() > runState->n_consumed) {
        runState->embd.push_back(runState->embd_inp[runState->n_consumed]);
        runState->last_n_tokens.erase(runState->last_n_tokens.begin());
        runState->last_n_tokens.push_back(runState->embd_inp[runState->n_consumed]);
        ++runState->n_consumed;
        if ((int)runState->embd.size() >= params.batchSize) {
          break;
        }
      }
    }

    // return text results
    if (!self.isCancelled && !ignoreOutputtedTokens) {
      for (auto id : runState->embd) {
        NSString *token = [NSString stringWithCString:llama_token_to_str(_context.ctx, id) encoding:NSUTF8StringEncoding];
        [self _postEvent:[_CameLLMPredictionEvent outputTokenWithToken:token]];
      }
    }

    // if not currently processing queued inputs check if we should inject the prompt.
    if ((int)runState->embd_inp.size() <= runState->n_consumed) {
      // check for reverse prompt
      if (params.antiprompts.count > 0) {
        std::string last_output;
        for (auto id : runState->last_n_tokens) {
          last_output += llama_token_to_str(_context.ctx, id);
        }

        runState->is_antiprompt = false;
        // Check if each of the reverse prompts appears at the end of the output.
        for (NSString *antiprompt in params.antiprompts) {
          const char *cString = [antiprompt cStringUsingEncoding:NSUTF8StringEncoding];
          NSUInteger length = [antiprompt lengthOfBytesUsingEncoding:NSUTF8StringEncoding];
          if (last_output.find(cString, last_output.length() - length, length) != std::string::npos) {
            runState->is_antiprompt = true;
            return YES;
          }
        }
      }

      if (runState->n_past > 0 && needsToInjectPrompt) {
        std::string buffer;

        // instruct mode: insert instruction prefix
        if (hasPrefix && !runState->is_antiprompt) {
          runState->n_consumed = (int)runState->embd_inp.size();
          runState->embd_inp.insert(runState->embd_inp.end(), inp_pfx.begin(), inp_pfx.end());
        }

        // inject the prompt.
        std::string prompt([_prompt cStringUsingEncoding:NSUTF8StringEncoding]);
        std::vector<llama_token> prompt_inp;
        if (![LlamaOperationUtils tokenizeString:prompt with:_context into:prompt_inp addBeginningOfSequence:false outError:&tokenizeError]) {
          [self _postEvent:[_CameLLMPredictionEvent failedWithError:tokenizeError]];
          return NO;
        }
        runState->embd_inp.insert(runState->embd_inp.end(), prompt_inp.begin(), prompt_inp.end());

        // instruct mode: insert response suffix
        if (hasSuffix) {
          runState->embd_inp.insert(runState->embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
        }

        runState->n_remain -= prompt_inp.size();

        // do not output this again
        ignoreOutputtedTokens = YES;
      }

      if (runState->n_past > 0) {
        needsToInjectPrompt = NO;
      }
    }

    // end of text token
    if (!runState->embd.empty() && runState->embd.back() == llama_token_eos()) {
      return YES;
    }

    // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
    if (runState->n_remain <= 0 && params.numberOfTokens != -1) {
      runState->n_remain = params.numberOfTokens;
      return YES;
    }
  }

  // should be a noop
  return YES;
}

#pragma mark - Private

- (void)_postEvent:(_CameLLMPredictionEvent *)event
{
  if (_eventHandler != NULL) {
    _eventHandler(event);
  }
}

@end
