// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "MambaASRRunner",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "MambaASRRunner", targets: ["MambaASRRunner"])
    ],
    dependencies: [
    ],
    targets: [
        .executableTarget(
            name: "MambaASRRunner",
            dependencies: []
        )
    ]
)
