//
//  LlamaSessionParams.h
//  llama
//
//  Created by Alex Rozanski on 13/03/2023.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSUInteger, _LlamaSessionMode) {
  _LlamaSessionModeRegular = 0,
  _LlamaSessionModeInstructional
};

__attribute__((visibility("default")))
@interface _LlamaSessionParams : NSObject

// model in gpt_params
@property (nonatomic, readonly, copy) NSString *modelPath;

// replaces instruct in gpt_params
@property (nonatomic, readonly, assign) _LlamaSessionMode mode;

// seed in gpt_params
@property (nonatomic, assign) int32_t seed;
// n_threads in gpt_params
@property (nonatomic, assign) int32_t numberOfThreads;
// n_predict in gpt_params
@property (nonatomic, assign) int32_t numberOfTokens;
// n_gpu_layers in gpt_params
@property (nonatomic, assign) int32_t numberOfLayers;
// n_ctx in gpt_params
@property (nonatomic, assign) int32_t contextSize;
// n_batch in gpt_params
@property (nonatomic, assign) int32_t batchSize;
// repeat_last_n in gpt_params
@property (nonatomic, assign) int32_t lastNTokensToPenalize;
// n_keep in gpt_params
@property (nonatomic, assign) int32_t numberOfTokensToKeepFromInitialPrompt;

// top_k in gpt_params
@property (nonatomic, assign) int32_t topK;
// top_p in gpt_params
@property (nonatomic, assign) float topP;
// tfs_z in gpt_params
@property (nonatomic, assign) float tfsZ;
// typical_p in gpt_params
@property (nonatomic, assign) float typicalP;
// temp in gpt_params
@property (nonatomic, assign) float temp;
// repeat_penalty in gpt_params
@property (nonatomic, assign) float repeatPenalty;
// frequency_penalty in gpt_params
@property (nonatomic, assign) float frequencyPenalty;
// presence_penalty in gpt_params
@property (nonatomic, assign) float presencePenalty;
// mirostat in gpt_params
@property (nonatomic, assign) int mirostat;
// mirostat_tau in gpt_params
@property (nonatomic, assign) float mirostatTau;
// mirostat_eta in gpt_params
@property (nonatomic, assign) float mirostatEta;

@property (nullable, copy) NSArray<NSString *> *antiprompts;

// lora_adapter in gpt_params
@property (nullable, copy) NSString *loraAdapter;
// lora_base in gpt_params
@property (nullable, copy) NSString *loraBase;

// memory_f16 in gpt_params
@property (nonatomic, assign) BOOL useF16Memory;
// use_mmap in gpt_params
@property (nonatomic, assign) BOOL useMmap;
// use_mlock in gpt_params
@property (nonatomic, assign) BOOL useMlock;
// penalize_nl in gpt_params
@property (nonatomic, assign) BOOL penalizeNewLines;

// Support for other model types
@property (nonatomic, nullable, copy) NSString *initialPrompt;
@property (nonatomic, nullable, copy) NSString *promptPrefix;
@property (nonatomic, nullable, copy) NSString *promptSuffix;

// Convenience properties
@property (nonatomic, readonly) BOOL isInstructional;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)defaultParamsWithModelPath:(NSString *)modelPath mode:(_LlamaSessionMode)mode;

@end

NS_ASSUME_NONNULL_END
