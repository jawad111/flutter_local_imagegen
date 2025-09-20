import Flutter
import UIKit
import ImagePlayground

public class FlutterLocalImagegenPlugin: NSObject, FlutterPlugin {
    
    private let playground = Playground()
    
    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(name: "flutter_local_imagegen", binaryMessenger: registrar.messenger())
        let instance = FlutterLocalImagegenPlugin()
        registrar.addMethodCallDelegate(instance, channel: channel)
    }

    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "generateImage":
            guard let args = call.arguments as? [String: Any],
                  let prompt = args["prompt"] as? String else {
                result(FlutterError(code: "BAD_ARGS", message: "Missing prompt string", details: nil))
                return
            }
            generateImage(prompt: prompt, result: result)
            
        default:
            result(FlutterMethodNotImplemented)
        }
    }
    
    private func generateImage(prompt: String, result: @escaping FlutterResult) {
        Task {
            do {
                // Generate image using ImagePlayground
                let image = try await playground.generateImage(prompt: prompt)
                
                // Convert UIImage → Base64 PNG
                if let data = image.pngData() {
                    let base64String = data.base64EncodedString()
                    result(base64String)
                } else {
                    result(FlutterError(code: "IMAGE_ERROR", message: "Failed to convert image to PNG", details: nil))
                }
            } catch {
                result(FlutterError(code: "GEN_ERROR", message: error.localizedDescription, details: nil))
            }
        }
    }
}
