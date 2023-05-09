// swift-tools-version: 5.5

import PackageDescription

let package = Package(
  name: "CameLLMLlama",
  platforms: [
    .macOS(.v10_15),
    .iOS(.v13),
  ],
  products: [
    .library(
      name: "CameLLMLlama",
      targets: ["CameLLMLlama"]),
  ],
  dependencies: [
    .package(url: "https://github.com/CameLLM/CameLLM", branch: "main"),
    .package(url: "https://github.com/CameLLM/CameLLM-Plugin-Harness", branch: "main"),
    .package(url: "https://github.com/CameLLM/CameLLM-Common", branch: "main"),
    .package(url: "https://github.com/alexrozanski/Coquille.git", from: "0.3.0")
  ],
  targets: [
    .target(
      name: "CameLLMLlama",
      dependencies: [
        .product(name: "CameLLM", package: "CameLLM"),
        .product(name: "CameLLMPluginHarness", package: "CameLLM-Plugin-Harness"),
        .product(name: "CameLLMCommon", package: "CameLLM-Common"),
        "CameLLMLlamaObjCxx",
        "Coquille"
      ],
      resources: [
        .copy("resources/convert.py"),
        .copy("resources/convert-pth-to-ggml.py"),
        .copy("resources/convert-lora-to-ggml.py"),
      ]
    ),
    .target(
      name: "CameLLMLlamaObjCxx",
      dependencies: [
        .product(name: "CameLLMObjCxx", package: "CameLLM"),
        .product(name: "CameLLMPluginHarnessObjCxx", package: "CameLLM-Plugin-Harness"),
        .product(name: "CameLLMCommonObjCxx", package: "CameLLM-Common"),
      ],
      cSettings: [.unsafeFlags(["-Wno-shorten-64-to-32", "-fvisibility=hidden", "-fmodules", "-fcxx-modules"]), .define("GGML_USE_ACCELERATE")],
      cxxSettings: [
        .headerSearchPath("cpp"),
        .headerSearchPath("session/operations"),
        .headerSearchPath("internal")
      ],
      linkerSettings: [
        .linkedFramework("Accelerate")
      ]
    )
  ],
  cLanguageStandard: .gnu11,
  cxxLanguageStandard: .cxx11
)
