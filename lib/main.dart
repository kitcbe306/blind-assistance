import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:path_provider/path_provider.dart';

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: BlindAssistanceApp(),
    );
  }
}

class BlindAssistanceApp extends StatefulWidget {
  const BlindAssistanceApp({super.key});

  @override
  _BlindAssistanceAppState createState() => _BlindAssistanceAppState();
}

class _BlindAssistanceAppState extends State<BlindAssistanceApp> {
  late CameraController controller;
  bool isDetecting = false;
  late Interpreter interpreter;
  late List<String> labels;
  FlutterTts flutterTts = FlutterTts();

  @override
  void initState() {
    super.initState();
    initModel();
    initCamera();
  }

  Future<void> initModel() async {
    try {
      // Load model
      final modelPath = await _getModel('yolov5s-int8.tflite');
      interpreter = await Interpreter.fromFile(modelPath);

      // Load labels
      final labelPath = await _getModel('labels.txt');
      labels = await File(labelPath).readAsLines();
      
      print("Model and labels loaded successfully");
    } catch (e) {
      print("Failed to load model: $e");
    }
  }

  Future<String> _getModel(String filename) async {
    final dir = await getApplicationDocumentsDirectory();
    return '${dir.path}/$filename';
  }

  Future<void> initCamera() async {
    controller = CameraController(cameras[0], ResolutionPreset.medium);
    await controller.initialize();
    controller.startImageStream((CameraImage img) {
      if (!isDetecting) {
        isDetecting = true;
        detectObjects(img);
      }
    });
  }

  Future<void> detectObjects(CameraImage img) async {
    try {
      // Preprocess image (YOLOv5 expects 320x320)
      final input = _imageToByteList(img, 320);

      // Run inference
      final output = List.filled(1 * 25200 * 6, 0).reshape([1, 25200, 6]);
      interpreter.run(input, output);

      // Process detections
      final detections = _processOutput(output);
      if (detections.isNotEmpty) {
        final bestDetection = detections.first;
        await flutterTts.speak(
            "${labels[bestDetection.classIndex]} detected with ${(bestDetection.score * 100).toStringAsFixed(0)}% confidence");
      }
    } catch (e) {
      print("Detection error: $e");
    } finally {
      isDetecting = false;
    }
  }

  Uint8List _imageToByteList(CameraImage image, int inputSize) {
    // Implement image preprocessing (RGB, normalized, resized)
    // ... (See preprocessing steps below)
    return convertedBytes;
  }

  List<Detection> _processOutput(List<dynamic> output) {
    // Implement YOLOv5 output processing (NMS, thresholding)
    // ... (See output processing below)
    return filteredDetections;
  }

  @override
  void dispose() {
    controller.dispose();
    interpreter.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!controller.value.isInitialized) {
      return Container();
    }
    return Scaffold(
      body: CameraPreview(controller),
    );
  }
}

class Detection {
  final int classIndex;
  final double score;
  final Rect rect;

  Detection(this.classIndex, this.score, this.rect);
}