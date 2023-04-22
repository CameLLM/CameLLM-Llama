//
//  PredictionPayload.swift
//  
//
//  Created by Alex Rozanski on 22/04/2023.
//

import Foundation

// Holds information about queued prediction.
struct PredictionPayload {
  let identifier: String
  let prompt: String
  let startHandler: () -> Void
  let tokenHandler: (String) -> Void
  let completionHandler: () -> Void
  let cancelHandler: () -> Void
  let failureHandler: (Error) -> Void
  let handlerQueue: DispatchQueue
}
