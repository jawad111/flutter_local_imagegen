import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'flutter_local_imagegen_platform_interface.dart';

/// An implementation of [FlutterLocalImagegenPlatform] that uses method channels.
class MethodChannelFlutterLocalImagegen extends FlutterLocalImagegenPlatform {
  /// The method channel used to interact with the native platform.
  @visibleForTesting
  final methodChannel = const MethodChannel('flutter_local_imagegen');

  /// Generates an image from a text prompt on-device using the native API.
  @override
  Future<Uint8List?> generateImage(String prompt) async {
    try {
      final base64Image = await methodChannel.invokeMethod<String>(
        'generateImage',
        {'prompt': prompt},
      );

      if (base64Image == null) return null;

      return base64Decode(base64Image);
    } on PlatformException catch (e) {
      debugPrint('Image generation failed: ${e.message}');
      return null;
    }
  }
}
