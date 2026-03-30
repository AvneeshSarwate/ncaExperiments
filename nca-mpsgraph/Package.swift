// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "NCATrainer",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "NCATrainer",
            exclude: ["nca_kernels.metal"]
        )
    ]
)
