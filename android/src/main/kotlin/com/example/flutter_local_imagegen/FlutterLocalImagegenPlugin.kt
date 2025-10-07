package com.example.flutter_local_imagegen

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Handler
import android.os.Looper
import android.util.Base64
import android.util.Log
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
  private val TAG = "FlutterLocalImagegen"

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
            Log.e(TAG, "Image generation failed", t)
            mainHandler.post { result.error("GEN_FAIL", t.stackTraceToString(), null) }
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
    val cond = encodeText(prompt)
    val uncond = encodeText("")

    // 256x256 latent (4, 32, 32)
    val latentShape = intArrayOf(1, 4, 32, 32)
    var latents = generateGaussianNoise(latentShape, seed)

    val guidanceScale = 7.5f
    val scheduler = EulerScheduler(numInferenceSteps)
    // Start from correct noise scale
    latents = scaleLatents(latents, scheduler.sigmas.first())
    for (i in scheduler.indices) {
      val tIdx = scheduler.timesteps[i]
      val sigma = scheduler.sigmas[i]
      val modelInput = scaleModelInput(latents, sigma)
      val noisePred = runUnetNoisePrediction(modelInput, tIdx, cond, uncond, guidanceScale)
      latents = scheduler.step(env, latents, noisePred, i)
      modelInput.close()
    }

    val bmp = decodeLatentsToBitmap(latents)
    cond.close()
    uncond.close()
    return bmp
  }

  // -------- Text encoder --------

  private fun encodeText(prompt: String): OnnxTensor {
    val session = textEncoderSession ?: error("Text encoder not initialized")
    if (vocab == null || bpeRanks == null) loadTokenizer()
    val maxLen = 77
    val tokens = tokenizeToIds(prompt, maxLen)

    val inputIds = IntArray(maxLen) { tokens[it].toInt() }
    val attn = IntArray(maxLen) { if (inputIds[it] != 0) 1 else 0 }

    val inputIdsTensor = OnnxTensor.createTensor(
      env, java.nio.IntBuffer.wrap(inputIds), longArrayOf(1, maxLen.toLong())
    )

    val inputNames = session.inputInfo.keys.map { it }
    Log.d(TAG, "Text encoder expects inputs: $inputNames")

    val feeds = HashMap<String, OnnxTensor>(2)
    when {
      inputNames.contains("input_ids") -> feeds["input_ids"] = inputIdsTensor
      inputNames.contains("input") -> feeds["input"] = inputIdsTensor
      else -> feeds[inputNames.first()] = inputIdsTensor
    }

    var attnTensor: OnnxTensor? = null
    if (inputNames.contains("attention_mask")) {
      attnTensor = OnnxTensor.createTensor(
        env, java.nio.IntBuffer.wrap(attn), longArrayOf(1, maxLen.toLong())
      )
      feeds["attention_mask"] = attnTensor
    }

    val outputs = session.run(feeds)

    val hidden = outputs[0].value as Array<Array<FloatArray>> // [1, 77, hidden]
    outputs.close()
    inputIdsTensor.close()
    attnTensor?.close()

    val shape = longArrayOf(1, hidden[0].size.toLong(), hidden[0][0].size.toLong())
    val fb = floatBufferFrom3D(hidden)
    return OnnxTensor.createTensor(env, fb, shape)
  }

  // -------- U-Net step --------

  private fun runUnetNoisePrediction(
    latents: OnnxTensor,
    timestep: Int,
    cond: OnnxTensor,
    uncond: OnnxTensor,
    guidanceScale: Float
  ): OnnxTensor {
    val session = unetSession ?: error("U-Net not initialized")

    // Prepare batch=2 latents by concatenating along batch dim: [uncond, cond]
    val latentsArray = tensorTo4DArray(latents)
    val latentsB2 = arrayOf(latentsArray[0], latentsArray[0])
    val latentsB2Tensor = float4DTensor(latentsB2)

    // Concatenate hidden states along batch: [uncond, cond]
    val condArr = tensorTo3DArray(cond)
    val uncondArr = tensorTo3DArray(uncond)
    val encB2 = arrayOf(uncondArr[0], condArr[0])
    val encB2Tensor = float3DTensor(encB2)

    val timestepTensor = OnnxTensor.createTensor(
      env, java.nio.IntBuffer.wrap(intArrayOf(timestep)), longArrayOf(1)
    )

    val inputNames = session.inputInfo.keys.toList()
    // Map common aliases
    val sampleKey = when {
      inputNames.contains("sample") -> "sample"
      inputNames.contains("x") -> "x"
      else -> inputNames.first()
    }
    val timestepKey = when {
      inputNames.contains("timestep") -> "timestep"
      inputNames.contains("t") -> "t"
      else -> inputNames.first { it != sampleKey }
    }
    val contextKey = when {
      inputNames.contains("encoder_hidden_states") -> "encoder_hidden_states"
      inputNames.contains("context") -> "context"
      else -> inputNames.first { it != sampleKey && it != timestepKey }
    }

    val feeds = hashMapOf<String, OnnxTensor>()
    feeds[sampleKey] = latentsB2Tensor
    feeds[timestepKey] = timestepTensor
    feeds[contextKey] = encB2Tensor

    val outputs = session.run(feeds)

    val outB2 = outputs[0].value as Array<Array<Array<FloatArray>>> // [2,4,32,32]
    outputs.close()

    // CFG: eps = eps_uncond + s * (eps_cond - eps_uncond)
    val epsUncond = outB2[0]
    val epsCond = outB2[1]
    val cfg = Array(epsUncond.size) { c ->
      Array(epsUncond[0].size) { y ->
        FloatArray(epsUncond[0][0].size) { x ->
          val u = epsUncond[c][y][x]
          val v = epsCond[c][y][x]
          (u + guidanceScale * (v - u))
        }
      }
    }

    timestepTensor.close()
    latentsB2Tensor.close()
    encB2Tensor.close()
    // Do not close 'latents' here; caller manages its lifecycle
    return float4DTensor(arrayOf(cfg)) // shape [1,4,32,32]
  }

  // -------- VAE decode --------

  private fun decodeLatentsToBitmap(latents: OnnxTensor): Bitmap {
    val session = vaeDecoderSession ?: error("VAE decoder not initialized")
    // SD convention: scale latents by 1 / 0.18215 before VAE decode
    val scaled = scaleLatents(latents, (1.0f / 0.18215f))
    val outputs = session.run(mapOf("latent_sample" to scaled))
    val image = outputs[0].value as Array<Array<Array<FloatArray>>> // [1,3,H,W], range [-1,1]
    outputs.close()
    latents.close()
    scaled.close()

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

  private fun scaleLatents(latents: OnnxTensor, scale: Float): OnnxTensor {
    val arr = tensorTo4DArray(latents)[0]
    val scaled = Array(arr.size) { c ->
      Array(arr[0].size) { y ->
        FloatArray(arr[0][0].size) { z ->
          arr[c][y][z] * scale
        }
      }
    }
    return float4DTensor(arrayOf(scaled))
  }

  // -------- ORT / tokenizer init --------

  private fun ensureOrtInit() {
    if (env != null) return
    env = OrtEnvironment.getEnvironment()
  }

  private fun ensureModelsLoaded() {
    if (textEncoderSession != null && unetSession != null && vaeDecoderSession != null) return

    val so = SessionOptions().apply {
      try {
        // Prefer NNAPI on Android devices supporting it, fallback to CPU
        addNnapi()
      } catch (_: Throwable) {
        // NNAPI EP not available, proceed with CPU EP
      }
      setOptimizationLevel(SessionOptions.OptLevel.ALL_OPT)
      setIntraOpNumThreads(4)
      setInterOpNumThreads(4)
    }

    val textEncFile = ensureAssetToFile(appContext, TEXT_ENCODER_ASSET)
    val unetFile = ensureAssetToFile(appContext, UNET_ASSET)
    val vaeFile = ensureAssetToFile(appContext, VAE_DECODER_ASSET)

    try {
      Log.d(TAG, "Creating ORT sessions with files: text=${textEncFile.absolutePath}, unet=${unetFile.absolutePath}, vae=${vaeFile.absolutePath}")
      textEncoderSession = env!!.createSession(textEncFile.absolutePath, so)
      unetSession = env!!.createSession(unetFile.absolutePath, so)
      vaeDecoderSession = env!!.createSession(vaeFile.absolutePath, so)
    } catch (t: Throwable) {
      Log.e(TAG, "Failed to create ORT sessions", t)
      throw t
    }

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
    val vjson = openPluginAsset(VOCAB_ASSET).bufferedReader().use { it.readText() }
    val jo = JSONObject(vjson)
    val map = HashMap<String, Long>(jo.length())
    jo.keys().forEach { k ->
      val id = jo.getInt(k)
      map[k] = id.toLong()
    }
    vocab = map

    try {
      val sjson = openPluginAsset(SPECIAL_TOKENS_ASSET).bufferedReader().use { it.readText() }
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
    openPluginAsset("tokenizer/merges.txt").use { ins ->
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

  private fun openPluginAsset(assetPath: String) =
    // Prefer plugin-scoped asset path first, then app-level assets
    appContext.assets.open(
      flutterAssets?.getAssetFilePathByName("packages/flutter_local_imagegen/assets/$assetPath")
        ?: flutterAssets?.getAssetFilePathByName("assets/$assetPath")
        ?: run {
          val fallback = "packages/flutter_local_imagegen/assets/$assetPath"
          Log.w(TAG, "Fallback asset path for $assetPath -> $fallback")
          fallback
        }
    )

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
    val resolved =
      flutterAssets?.getAssetFilePathByName("packages/flutter_local_imagegen/assets/$assetPath")
        ?: flutterAssets?.getAssetFilePathByName("assets/$assetPath")
        ?: flutterAssets?.getAssetFilePathByName(assetPath)
        ?: "packages/flutter_local_imagegen/assets/$assetPath"
    Log.d(TAG, "Resolving asset '$assetPath' -> '$resolved' -> '${outFile.absolutePath}'")
    try {
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
      if (outFile.length() == 0L) error("Copied zero-byte asset: $assetPath from $resolved")
    } catch (t: Throwable) {
      Log.e(TAG, "Failed to open/copy asset '$assetPath' resolved as '$resolved'", t)
      throw t
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

  private fun tensorTo4DArray(t: OnnxTensor): Array<Array<Array<FloatArray>>> {
    @Suppress("UNCHECKED_CAST")
    return t.value as Array<Array<Array<FloatArray>>>
  }

  private fun float3DTensor(arr: Array<Array<FloatArray>>): OnnxTensor {
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
    return OnnxTensor.createTensor(env, fb, longArrayOf(d0.toLong(), d1.toLong(), d2.toLong()))
  }

  private fun tensorTo3DArray(t: OnnxTensor): Array<Array<FloatArray>> {
    @Suppress("UNCHECKED_CAST")
    return t.value as Array<Array<FloatArray>>
  }

  private fun generateGaussianNoise(shape: IntArray, seed: Long): OnnxTensor {
    val total = shape.fold(1) { acc, i -> acc * i }
    val rnd = java.util.Random(seed)
    val fb = ByteBuffer.allocateDirect(total * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    repeat(total) { fb.put(rnd.nextGaussian().toFloat()) }
    fb.rewind()
    return OnnxTensor.createTensor(env, fb, shape.map { it.toLong() }.toLongArray())
  }

  private class EulerScheduler(private val steps: Int) {
    private val trainingTimesteps = 1000
    val timesteps: IntArray
    val sigmas: FloatArray
    val indices: IntArray

    init {
      val (ts, s) = buildSchedule(steps, trainingTimesteps)
      timesteps = ts
      sigmas = s
      indices = IntArray(steps) { it }
    }

    fun step(env: OrtEnvironment?, latents: OnnxTensor, noisePred: OnnxTensor, i: Int): OnnxTensor {
      val sigmaT = sigmas[i]
      val sigmaNext = if (i < sigmas.size - 1) sigmas[i + 1] else 0.0f
      val dt = sigmaNext - sigmaT
      val x = (latents.value as Array<Array<Array<FloatArray>>>)[0]
      val n = (noisePred.value as Array<Array<Array<FloatArray>>>)[0]
      val updated = Array(x.size) { c ->
        Array(x[0].size) { y ->
          FloatArray(x[0][0].size) { z ->
            // Euler update: x_{t+1} = x_t + dt * d, where d ≈ -eps
            x[c][y][z] + dt * (-n[c][y][z])
          }
        }
      }
      latents.close()
      noisePred.close()
      val byteCount = updated.size * updated[0].size * updated[0][0].size * 4
      val fb = ByteBuffer.allocateDirect(byteCount).order(ByteOrder.nativeOrder()).asFloatBuffer()
      for (c in updated.indices) for (y in updated[c].indices) fb.put(updated[c][y])
      fb.rewind()
      return OnnxTensor.createTensor(env, fb,
        longArrayOf(1, updated.size.toLong(), updated[0].size.toLong(), updated[0][0].size.toLong()))
    }

    private fun buildSchedule(steps: Int, trainSteps: Int): Pair<IntArray, FloatArray> {
      // Diffusers v1 scaled_linear beta schedule → alphas_cumprod → sigmas
      val betaStart = 0.00085
      val betaEnd = 0.012
      val betas = DoubleArray(trainSteps) { idx ->
        val t = idx.toDouble() / (trainSteps - 1).toDouble()
        val beta = (Math.pow(betaStart, 0.5) + t * (Math.pow(betaEnd, 0.5) - Math.pow(betaStart, 0.5)))
        beta * beta
      }
      val alphas = DoubleArray(trainSteps) { 1.0 - betas[it] }
      val alphasCumprod = DoubleArray(trainSteps)
      var prod = 1.0
      for (i in 0 until trainSteps) { prod *= alphas[i]; alphasCumprod[i] = prod }
      val sigmasTrain = DoubleArray(trainSteps) { i ->
        Math.sqrt((1.0 - alphasCumprod[i]) / alphasCumprod[i])
      }

      // Choose step indices linearly in [0, trainSteps-1], descending
      val tIdx = IntArray(steps) { k ->
        val v = (trainSteps - 1) - Math.floor(k * (trainSteps - 1).toDouble() / (steps - 1).toDouble()).toInt()
        v.coerceIn(0, trainSteps - 1)
      }
      val s = FloatArray(steps) { k -> sigmasTrain[tIdx[k]].toFloat() }
      return Pair(tIdx, s)
    }
  }

  private fun scaleModelInput(latents: OnnxTensor, sigma: Float): OnnxTensor {
    // scale_model_input(x, sigma) = x / sqrt(sigma^2 + 1)
    val x = tensorTo4DArray(latents)[0]
    val scale = (1.0f / Math.sqrt((sigma * sigma + 1.0f).toDouble())).toFloat()
    val scaled = Array(x.size) { c ->
      Array(x[0].size) { y ->
        FloatArray(x[0][0].size) { z ->
          x[c][y][z] * scale
        }
      }
    }
    return float4DTensor(arrayOf(scaled))
  }
}

