import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_local_imagegen/flutter_local_imagegen.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: ImageGenExample(),
    );
  }
}

class ImageGenExample extends StatefulWidget {
  const ImageGenExample({super.key});

  @override
  State<ImageGenExample> createState() => _ImageGenExampleState();
}

class _ImageGenExampleState extends State<ImageGenExample> {
  final TextEditingController _promptController = TextEditingController();
  Uint8List? _generatedImage;
  bool _isLoading = false;

  Future<void> _generateImage() async {
    final prompt = _promptController.text.trim();
    if (prompt.isEmpty) return;

    setState(() {
      _isLoading = true;
      _generatedImage = null;
    });

    try {
      final imageBytes = await FlutterLocalImagegen.generateImage(prompt);
      setState(() {
        _generatedImage = imageBytes;
      });
    } catch (e) {
      debugPrint('Error generating image: $e');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to generate image: $e')),
      );
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Local Image Generator")),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            TextField(
              controller: _promptController,
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                labelText: "Enter prompt",
              ),
            ),
            const SizedBox(height: 12),
            ElevatedButton(
              onPressed: _isLoading ? null : _generateImage,
              child: const Text("Generate"),
            ),
            const SizedBox(height: 20),
            if (_isLoading) const CircularProgressIndicator(),
            if (_generatedImage != null) ...[
              const SizedBox(height: 20),
              Expanded(
                child: Image.memory(
                  _generatedImage!,
                  fit: BoxFit.contain,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
