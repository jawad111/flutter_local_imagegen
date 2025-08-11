import 'dart:typed_data';
import 'flutter_local_imagegen_platform_interface.dart';

class FlutterLocalImagegen {
  /// Generates an image from the given text prompt.
  /// Returns the image bytes (PNG format) or null if generation fails.
  static Future<Uint8List?> generateImage(String prompt) {
    return FlutterLocalImagegenPlatform.instance.generateImage(prompt);
  }
}
