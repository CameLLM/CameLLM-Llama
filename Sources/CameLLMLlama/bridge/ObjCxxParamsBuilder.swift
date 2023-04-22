//
//  ObjCxxParamsBuilder.swift
//
//
//  Created by Alex Rozanski on 22/04/2023.
//

import Foundation
import CameLLMLlamaObjCxx

protocol ObjCxxParamsBuilder {
  func build(for modelURL: URL) -> _LlamaSessionParams
}
