package com.example.flutter_local_imagegen

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Handler
import android.os.Looper
import android.util.Base64
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtSession.SessionOptions
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.Executors
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.Locale

/** FlutterLocalImagegenPlugin */
class FlutterLocalImagegenPlugin : FlutterPlugin, MethodCallHandler {
  private lateinit var channel: MethodChannel
  private lateinit var appContext: Context
  private val executor = Executors.newSingleThreadExecutor()
  private val mainHandler = Handler(Looper.getMainLooper())
  private var flutterAssets: FlutterPlugin.FlutterAssets? = null

  // Assets
  private val TEXT_ENCODER_ASSET = "text_encoder/model.ort"
  private val UNET_ASSET = "unet/model.ort"
  private val VAE_DECODER_ASSET = "vae_decoder/model.ort"
  private val VOCAB_ASSET = "tokenizer/vocab.json"
  private val SPECIAL_TOKENS_ASSET = "tokenizer/special_tokens_map.json"

  // ORT
  private var env: OrtEnvironment? = null
  private var textEncoderSession: OrtSession? = null
  private var unetSession: OrtSession? = null
  private var vaeDecoderSession: OrtSession? = null

  // Tokenizer
  private var vocab: Map<String, Long>? = null
  private var bosTokenId: Long? = null
  private var eosTokenId: Long? = null
  private var unkTokenId: Long = 0L
  private var bpeRanks: Map<Pair<String, String>, Int>? = null

  override fun onAttachedToEngine(flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
    appContext = flutterPluginBinding.applicationContext
    channel = MethodChannel(flutterPluginBinding.binaryMessenger, "flutter_local_imagegen")
    flutterAssets = flutterPluginBinding.flutterAssets
    channel.setMethodCallHandler(this)
  }

  override fun onMethodCall(call: MethodCall, result: Result) {
    when (call.method) {
      "generateImage" -> {
        val prompt = call.argument<String>("prompt") ?: ""
        val steps = (call.argument<Int>("steps") ?: 20).coerceIn(1, 100)
        val seed = call.argument<Long>("seed") ?: 42L
        executor.execute {
          try {
            ensureOrtInit()
            ensureModelsLoaded()
            val bmp = generateImageFromText(prompt, steps, seed)
            val b64 = encodeBitmapToBase64(bmp)
            mainHandler.post { result.success(b64) }
          } catch (t: Throwable) {
            mainHandler.post { result.error("GEN_FAIL", t.message, null) }
          }
        }
      }
      else -> result.notImplemented()
    }
  }

  override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
    channel.setMethodCallHandler(null)
    executor.shutdown()
    closeSessions()
  }

  // -------- Pipeline --------

  private fun generateImageFromText(prompt: String, numInferenceSteps: Int, seed: Long): Bitmap {
    val clipHiddenStates = encodeText(prompt)

    // 512x512 latent (4, 64, 64)
    val latentShape = intArrayOf(1, 4, 64, 64)
    var latents = generateGaussianNoise(latentShape, seed)

    val scheduler = SimpleEulerScheduler(numInferenceSteps)
    for (t in scheduler.timesteps) {
      latents = runUnetStep(latents, t, clipHiddenStates)
      latents = scheduler.step(latents)
    }

    val bmp = decodeLatentsToBitmap(latents)
    clipHiddenStates.close()
    return bmp
  }

  // -------- Text encoder --------

  private fun encodeText(prompt: String): OnnxTensor {
    val session = textEncoderSession ?: error("Text encoder not initialized")
    if (vocab == null || bpeRanks == null) loadTokenizer()
    val maxLen = 77
    val tokens = tokenizeToIds(prompt, maxLen)

    val inputIds = LongArray(maxLen) { tokens[it] }
    val attn = LongArray(maxLen) { if (inputIds[it] != 0L) 1L else 0L }

    val inputIdsTensor = OnnxTensor.createTensor(
      env, java.nio.LongBuffer.wrap(inputIds), longArrayOf(1, maxLen.toLong())
    )
    val attnTensor = OnnxTensor.createTensor(
      env, java.nio.LongBuffer.wrap(attn), longArrayOf(1, maxLen.toLong())
    )

    val outputs = session.run(
      mapOf(
        "input_ids" to inputIdsTensor,
        "attention_mask" to attnTensor
      )
    )

    val hidden = outputs[0].value as Array<Array<FloatArray>> // [1, 77, hidden]
    outputs.close()
    inputIdsTensor.close()
    attnTensor.close()

    val shape = longArrayOf(1, hidden[0].size.toLong(), hidden[0][0].size.toLong())
    val fb = floatBufferFrom3D(hidden)
    return OnnxTensor.createTensor(env, fb, shape)
  }

  // -------- U-Net step --------

  private fun runUnetStep(
    latents: OnnxTensor,
    timestep: Int,
    clipHiddenStates: OnnxTensor
  ): OnnxTensor {
    val session = unetSession ?: error("U-Net not initialized")

    val timestepTensor = OnnxTensor.createTensor(
      env, FloatBuffer.wrap(floatArrayOf(timestep.toFloat())), longArrayOf(1)
    )

    val outputs = session.run(
      mapOf(
        "sample" to latents,
        "timestep" to timestepTensor,
        "encoder_hidden_states" to clipHiddenStates
      )
    )

    val out = outputs[0].value as Array<Array<Array<FloatArray>>> // same shape as latents
    outputs.close()
    timestepTensor.close()
    latents.close()
    return float4DTensor(out)
  }

  // -------- VAE decode --------

  private fun decodeLatentsToBitmap(latents: OnnxTensor): Bitmap {
    val session = vaeDecoderSession ?: error("VAE decoder not initialized")
    val outputs = session.run(mapOf("latent_sample" to latents))
    val image = outputs[0].value as Array<Array<Array<FloatArray>>> // [1,3,H,W], range [-1,1]
    outputs.close()
    latents.close()

    val h = image[0][0].size
    val w = image[0][0][0].size
    val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
    for (y in 0 until h) {
      for (x in 0 until w) {
        val r = clamp01((image[0][0][y][x] + 1f) / 2f)
        val g = clamp01((image[0][1][y][x] + 1f) / 2f)
        val b = clamp01((image[0][2][y][x] + 1f) / 2f)
        val color = Color.rgb((r * 255f).toInt(), (g * 255f).toInt(), (b * 255f).toInt())
        bmp.setPixel(x, y, color)
      }
    }
    return bmp
  }

  // -------- ORT / tokenizer init --------

  private fun ensureOrtInit() {
    if (env != null) return
    env = OrtEnvironment.getEnvironment()
  }

  private fun ensureModelsLoaded() {
    if (textEncoderSession != null && unetSession != null && vaeDecoderSession != null) return

    val so = SessionOptions().apply {
      setIntraOpNumThreads(4)
      setInterOpNumThreads(4)
    }

    val textEncFile = ensureAssetToFile(appContext, TEXT_ENCODER_ASSET)
    val unetFile = ensureAssetToFile(appContext, UNET_ASSET)
    val vaeFile = ensureAssetToFile(appContext, VAE_DECODER_ASSET)

    textEncoderSession = env!!.createSession(textEncFile.absolutePath, so)
    unetSession = env!!.createSession(unetFile.absolutePath, so)
    vaeDecoderSession = env!!.createSession(vaeFile.absolutePath, so)

    if (vocab == null) loadTokenizer()
  }

  private fun closeSessions() {
    textEncoderSession?.close()
    unetSession?.close()
    vaeDecoderSession?.close()
    env?.close()
    textEncoderSession = null
    unetSession = null
    vaeDecoderSession = null
    env = null
  }

  private fun loadTokenizer() {
    val vjson = appContext.assets.open(VOCAB_ASSET).bufferedReader().use { it.readText() }
    val jo = JSONObject(vjson)
    val map = HashMap<String, Long>(jo.length())
    jo.keys().forEach { k ->
      val id = jo.getInt(k)
      map[k] = id.toLong()
    }
    vocab = map

    try {
      val sjson = appContext.assets.open(SPECIAL_TOKENS_ASSET).bufferedReader().use { it.readText() }
      val so = JSONObject(sjson)
      fun extractToken(key: String): String? {
        if (!so.has(key)) return null
        val v = so.get(key)
        return when (v) {
          is String -> v
          is JSONObject -> v.optString("content", null)
          else -> null
        }
      }
      val bos = extractToken("bos_token")
      val eos = extractToken("eos_token")
      val unk = extractToken("unk_token")
      bosTokenId = bos?.let { map[it] }
      eosTokenId = eos?.let { map[it] }
      unkTokenId = unk?.let { map[it] } ?: unkTokenId
    } catch (_: Throwable) {
      // optional
    }

    // Load merges.txt into rank map
    val ranks = HashMap<Pair<String, String>, Int>()
    appContext.assets.open("tokenizer/merges.txt").use { ins ->
      BufferedReader(InputStreamReader(ins)).use { br ->
        var line: String?
        var rank = 0
        while (true) {
          line = br.readLine() ?: break
          if (line!!.isEmpty() || line!!.startsWith("#")) continue
          val parts = line!!.trim().split(" ")
          if (parts.size == 2) {
            ranks[Pair(parts[0], parts[1])] = rank
            rank += 1
          }
        }
      }
    }
    bpeRanks = ranks
  }

  // -------- BPE tokenizer (simplified) --------

  private fun tokenizeToIds(text: String, maxLen: Int): ArrayList<Long> {
    val result = ArrayList<Long>(maxLen)
    val v = vocab ?: error("vocab missing")
    val ranks = bpeRanks ?: error("merges missing")

    bosTokenId?.let { result.add(it) }

    val words = text.trim().lowercase(Locale.US).split(Regex("\\s+")).filter { it.isNotBlank() }
    outer@ for (w in words) {
      val bpeTokens = bpeEncode(w, v, ranks)
      for (t in bpeTokens) {
        if (result.size >= maxLen - 1) break@outer
        result.add(t)
      }
    }

    eosTokenId?.let { if (result.size < maxLen) result.add(it) }
    while (result.size < maxLen) result.add(0L)
    return result
  }

  private fun bpeEncode(word: String, vocab: Map<String, Long>, ranks: Map<Pair<String, String>, Int>): List<Long> {
    // Initialize with single-character tokens
    var tokens = word.map { it.toString() }
    if (tokens.isEmpty()) return listOf(unkTokenId)

    while (true) {
      val pairs = getPairs(tokens)
      if (pairs.isEmpty()) break

      var minPair: Pair<String, String>? = null
      var minRank = Int.MAX_VALUE
      for (p in pairs) {
        val r = ranks[p]
        if (r != null && r < minRank) {
          minRank = r
          minPair = p
        }
      }
      if (minPair == null) break

      val merged = ArrayList<String>(tokens.size)
      var i = 0
      while (i < tokens.size) {
        if (i < tokens.size - 1 && tokens[i] == minPair.first && tokens[i + 1] == minPair.second) {
          merged.add(tokens[i] + tokens[i + 1])
          i += 2
        } else {
          merged.add(tokens[i])
          i += 1
        }
      }
      tokens = merged
    }

    val out = ArrayList<Long>(tokens.size)
    for (t in tokens) {
      out.add(vocab[t] ?: unkTokenId)
    }
    return out
  }

  private fun getPairs(tokens: List<String>): Set<Pair<String, String>> {
    val pairs = HashSet<Pair<String, String>>()
    var prev = tokens[0]
    var i = 1
    while (i < tokens.size) {
      val cur = tokens[i]
      pairs.add(Pair(prev, cur))
      prev = cur
      i += 1
    }
    return pairs
  }

  // -------- Utils --------

  private fun encodeBitmapToBase64(bitmap: Bitmap): String {
    val buffer = java.io.ByteArrayOutputStream()
    bitmap.compress(Bitmap.CompressFormat.PNG, 100, buffer)
    val bytes = buffer.toByteArray()
    return Base64.encodeToString(bytes, Base64.NO_WRAP)
  }

  private fun ensureAssetToFile(context: Context, assetPath: String): File {
    val outFile = File(context.filesDir, assetPath)
    if (outFile.exists() && outFile.length() > 0) return outFile
    outFile.parentFile?.mkdirs()
    // Resolve Flutter asset path (works for app assets and plugin package assets)
    val resolved = flutterAssets?.getAssetFilePathByName(assetPath)
      ?: flutterAssets?.getAssetFilePathByName("packages/flutter_local_imagegen/$assetPath")
      ?: assetPath
    context.assets.open(resolved).use { input ->
      FileOutputStream(outFile).use { output ->
        val buf = ByteArray(8 * 1024)
        while (true) {
          val read = input.read(buf)
          if (read <= 0) break
          output.write(buf, 0, read)
        }
        output.flush()
      }
    }
    return outFile
  }

  private fun clamp01(v: Float): Float = when {
    v < 0f -> 0f
    v > 1f -> 1f
    else -> v
  }

  private fun floatBufferFrom3D(arr: Array<Array<FloatArray>>): FloatBuffer {
    val d0 = arr.size
    val d1 = arr[0].size
    val d2 = arr[0][0].size
    val fb = ByteBuffer.allocateDirect(d0 * d1 * d2 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    for (i in 0 until d0) {
      for (j in 0 until d1) {
        fb.put(arr[i][j])
      }
    }
    fb.rewind()
    return fb
  }

  private fun float4DTensor(arr: Array<Array<Array<FloatArray>>>): OnnxTensor {
    val d0 = arr.size
    val d1 = arr[0].size
    val d2 = arr[0][0].size
    val d3 = arr[0][0][0].size
    val fb = ByteBuffer.allocateDirect(d0 * d1 * d2 * d3 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    for (i in 0 until d0) {
      for (j in 0 until d1) {
        for (k in 0 until d2) {
          fb.put(arr[i][j][k])
        }
      }
    }
    fb.rewind()
    return OnnxTensor.createTensor(env, fb, longArrayOf(d0.toLong(), d1.toLong(), d2.toLong(), d3.toLong()))
  }

  private fun generateGaussianNoise(shape: IntArray, seed: Long): OnnxTensor {
    val total = shape.fold(1) { acc, i -> acc * i }
    val rnd = java.util.Random(seed)
    val fb = ByteBuffer.allocateDirect(total * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    repeat(total) { fb.put(rnd.nextGaussian().toFloat()) }
    fb.rewind()
    return OnnxTensor.createTensor(env, fb, shape.map { it.toLong() }.toLongArray())
  }

  private class SimpleEulerScheduler(steps: Int) {
    val timesteps: IntArray = IntArray(steps) { steps - 1 - it }
    fun step(latents: OnnxTensor): OnnxTensor {
      return latents
    }
  }
}
