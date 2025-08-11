// import 'package:flutter_test/flutter_test.dart';
// import 'package:flutter_local_imagegen/flutter_local_imagegen.dart';
// import 'package:flutter_local_imagegen/flutter_local_imagegen_platform_interface.dart';
// import 'package:flutter_local_imagegen/flutter_local_imagegen_method_channel.dart';
// import 'package:plugin_platform_interface/plugin_platform_interface.dart';

// class MockFlutterLocalImagegenPlatform
//     with MockPlatformInterfaceMixin
//     implements FlutterLocalImagegenPlatform {

//   @override
//   Future<String?> getPlatformVersion() => Future.value('42');
// }

// void main() {
//   final FlutterLocalImagegenPlatform initialPlatform = FlutterLocalImagegenPlatform.instance;

//   test('$MethodChannelFlutterLocalImagegen is the default instance', () {
//     expect(initialPlatform, isInstanceOf<MethodChannelFlutterLocalImagegen>());
//   });

//   test('getPlatformVersion', () async {
//     FlutterLocalImagegen flutterLocalImagegenPlugin = FlutterLocalImagegen();
//     MockFlutterLocalImagegenPlatform fakePlatform = MockFlutterLocalImagegenPlatform();
//     FlutterLocalImagegenPlatform.instance = fakePlatform;

//     expect(await flutterLocalImagegenPlugin.getPlatformVersion(), '42');
//   });
// }
