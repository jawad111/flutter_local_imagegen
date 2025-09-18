import Flutter
import UIKit
import ImagePlayground

public class FlutterLocalImagegenPlugin: NSObject, FlutterPlugin {
    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(
            name: "flutter_local_imagegen",
            binaryMessenger: registrar.messenger()
        )
        let instance = FlutterLocalImagegenPlugin()
        registrar.addMethodCallDelegate(instance, channel: channel)
    }

    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "generateImage":
            guard let args = call.arguments as? [String: Any],
                  let prompt = args["prompt"] as? String else {
                result(FlutterError(
                    code: "BAD_ARGS",
                    message: "Missing prompt string",
                    details: nil
                ))
                return
            }

            if #available(iOS 18.4, *) {
                generateImage(prompt: prompt, result: result)
            } else {
                result(FlutterError(
                    code: "IOS_VERSION",
                    message: "Requires iOS 18.4 or later",
                    details: nil
                ))
            }

        default:
            result(FlutterMethodNotImplemented)
        }
    }

    
    @available(iOS 18.4, *)
    private func generateImage(prompt: String, result: @escaping FlutterResult) {
        Task {
            do {
                let imageCreator = try await ImageCreator()
                let generationStyle = ImagePlaygroundStyle.animation
                
                var generatedImages: [CGImage] = []
                
                let images = imageCreator.images(
                    for: [.text(prompt)],
                    style: generationStyle,
                    limit: 3
                )
                
                for try await image in images {
                    generatedImages.append(image.cgImage)
                }
                
                if let firstImage = generatedImages.first {
                    let uiImage = UIImage(cgImage: firstImage)
                    if let data = uiImage.pngData() {
                        result(data.base64EncodedString())
                    } else {
                        result(FlutterError(
                            code: "IMAGE_ERROR",
                            message: "Failed to convert image to PNG",
                            details: nil
                        ))
                    }
                } else {
                    result(FlutterError(
                        code: "NO_IMAGE",
                        message: "No image generated",
                        details: nil
                    ))
                }
                
            } catch ImageCreator.Error.notSupported {
                result(FlutterError(
                    code: "NOT_SUPPORTED",
                    message: "Image creation not supported on this device.",
                    details: nil
                ))
            } catch {
                result(FlutterError(
                    code: "GEN_ERROR",
                    message: error.localizedDescription,
                    details: nil
                ))
            }
        }
    }
    
}
