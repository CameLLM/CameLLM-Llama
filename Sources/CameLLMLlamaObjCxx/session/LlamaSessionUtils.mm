//
//  LlamaSessionUtils.mm
//
//
//  Created by Alex Rozanski on 23/04/2023.
//

@import CameLLMObjCxx;

#import "LlamaSessionUtils.h"

#import "LlamaContext.h"
#import "LlamaContext+Internal.hh"

@implementation LlamaSessionUtils

+ (_SessionContext *)currentSessionContextWithLlamaContext:(LlamaContext *)context
{
  if (context.ctx == NULL || context.runState == NULL) {
    return nil;
  }

  NSMutableArray<_SessionContextToken *> *tokens = [[NSMutableArray alloc] init];

  for (auto &token : context.runState->last_n_tokens) {
    if (token == 0) { continue; }

    const char *cString = llama_token_to_str(context.ctx, token);
    NSString *string = [NSString stringWithCString:cString encoding:NSUTF8StringEncoding];
    if (!string) {
      string = @"";
    }
    [tokens addObject:[[_SessionContextToken alloc] initWithToken:token string:string]];
  }

  return [[_SessionContext alloc] initWithTokens:tokens];
}

@end
