import 'dart:typed_data';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';
import 'flutter_local_imagegen_method_channel.dart';

abstract class FlutterLocalImagegenPlatform extends PlatformInterface {
  /// Constructs a FlutterLocalImagegenPlatform.
  FlutterLocalImagegenPlatform() : super(token: _token);

  static final Object _token = Object();

  static FlutterLocalImagegenPlatform _instance =
      MethodChannelFlutterLocalImagegen();

  /// The default instance of [FlutterLocalImagegenPlatform] to use.
  ///
  /// Defaults to [MethodChannelFlutterLocalImagegen].
  static FlutterLocalImagegenPlatform get instance => _instance;

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [FlutterLocalImagegenPlatform] when
  /// they register themselves.
  static set instance(FlutterLocalImagegenPlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  /// Generates an image from the given text prompt.
  /// Returns the image bytes as Uint8List, or null on failure.
  Future<Uint8List?> generateImage(String prompt) {
    throw UnimplementedError('generateImage() has not been implemented.');
  }
}
