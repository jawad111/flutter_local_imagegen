package com.jawad.flutter_local_imagegen

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.os.Handler
import android.os.Looper
import android.util.Base64
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors

/** FlutterLocalImagegenPlugin */
class FlutterLocalImagegenPlugin : FlutterPlugin, MethodCallHandler {
  private lateinit var channel: MethodChannel
  private lateinit var appContext: Context
  private val executor = Executors.newSingleThreadExecutor()
  private val mainHandler = Handler(Looper.getMainLooper())

  override fun onAttachedToEngine(flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
    appContext = flutterPluginBinding.applicationContext
    channel = MethodChannel(flutterPluginBinding.binaryMessenger, "flutter_local_imagegen")
    channel.setMethodCallHandler(this)
  }

  override fun onMethodCall(call: MethodCall, result: Result) {
    when (call.method) {
      "generateImage" -> {
        val prompt = call.argument<String>("prompt") ?: ""
        executor.execute {
          try {
            val base64Image = generateDummyImage(prompt)
            mainHandler.post { result.success(base64Image) }
          } catch (t: Throwable) {
            mainHandler.post { result.error("GEN_FAIL", t.message, null) }
          }
        }
      }
      else -> result.notImplemented()
    }
  }

  private fun generateDummyImage(prompt: String): String {
    val width = 512
    val height = 512
    val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(bmp)

    // Background
    canvas.drawARGB(255, 50, 60, 90)

    // Draw the prompt text
    val paint = Paint().apply {
      color = 0xFFFFFFFF.toInt()
      textSize = 36f
      isAntiAlias = true
    }
    canvas.drawText(prompt.take(20), 40f, 100f, paint)

    // Encode bitmap as Base64 PNG
    val baos = ByteArrayOutputStream()
    bmp.compress(Bitmap.CompressFormat.PNG, 100, baos)
    val bytes = baos.toByteArray()
    return Base64.encodeToString(bytes, Base64.NO_WRAP)
  }

  override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
    channel.setMethodCallHandler(null)
    executor.shutdown()
  }
}
