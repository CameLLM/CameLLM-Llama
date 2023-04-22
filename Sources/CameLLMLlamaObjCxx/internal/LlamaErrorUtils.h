//
//  LlamaErrorUtils.h
//  
//
//  Created by Alex Rozanski on 23/04/2023.
//

#ifdef  __cplusplus
extern "C" {
#endif

NS_ASSUME_NONNULL_BEGIN

NSError *makeFailedToQuantizeErrorWithUnderlyingError(NSError *__nullable underlyingError);

NS_ASSUME_NONNULL_END

#ifdef  __cplusplus
}
#endif
