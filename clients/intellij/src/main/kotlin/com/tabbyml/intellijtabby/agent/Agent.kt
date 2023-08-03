package com.tabbyml.intellijtabby.agent

import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.intellij.execution.configurations.GeneralCommandLine
import com.intellij.execution.configurations.PathEnvironmentVariableUtil
import com.intellij.execution.process.KillableProcessHandler
import com.intellij.execution.process.ProcessAdapter
import com.intellij.execution.process.ProcessEvent
import com.intellij.execution.process.ProcessOutputTypes
import com.intellij.ide.plugins.PluginManagerCore
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.extensions.PluginId
import com.intellij.openapi.util.Key
import com.intellij.util.EnvironmentUtil
import com.intellij.util.io.BaseOutputReader
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.suspendCancellableCoroutine
import java.io.OutputStreamWriter

class Agent : ProcessAdapter() {
  private val logger = Logger.getInstance(Agent::class.java)
  private val gson = Gson()
  private val process: KillableProcessHandler
  private val streamWriter: OutputStreamWriter

  enum class Status {
    NOT_INITIALIZED,
    READY,
    DISCONNECTED,
    UNAUTHORIZED,
  }
  private val statusFlow = MutableStateFlow(Status.NOT_INITIALIZED)
  val status = statusFlow.asStateFlow()

  init {
    logger.info("Environment variables: PATH: ${EnvironmentUtil.getValue("PATH")}")

    val node = PathEnvironmentVariableUtil.findExecutableInPathOnAnyOS("node")
    if (node?.exists() == true) {
      logger.info("Node bin path: ${node.absolutePath}")
    } else {
      logger.error("Node bin not found")
      throw Error("Node bin not found")
    }

    val script =
      PluginManagerCore.getPlugin(PluginId.getId("com.tabbyml.intellij-tabby"))?.pluginPath?.resolve("node_scripts/tabby-agent.js")
        ?.toFile()
    if (script?.exists() == true) {
      logger.info("Node script path: ${script.absolutePath}")
    } else {
      logger.error("Node script not found")
      throw Error("Node script not found")
    }

    val cmd = GeneralCommandLine(node.absolutePath, script.absolutePath)
    process = object : KillableProcessHandler(cmd) {
      override fun readerOptions(): BaseOutputReader.Options {
        return BaseOutputReader.Options.forMostlySilentProcess()
      }
    }
    process.startNotify()
    process.addProcessListener(this)
    streamWriter = process.processInput.writer()
  }

  suspend fun initialize(): Boolean {
    return request("initialize", listOf(mapOf("client" to "intellij-tabby"))) // FIXME: correct client info
  }

  suspend fun updateConfig(): Boolean {
    return request("updateConfig", listOf(emptyMap<Any, Any>()))
  }

  data class CompletionRequest(
    val filepath: String,
    val language: String,
    val text: String,
    val position: Int,
  )

  data class CompletionResponse(
    val id: String,
    val choices: List<Choice>,
  ) {
    data class Choice(
      val index: Int,
      val text: String,
    )
  }

  suspend fun getCompletions(request: CompletionRequest): CompletionResponse? {
    return request("getCompletions", listOf(request))
  }

  fun close() {
    streamWriter.close()
    process.destroyProcess()
  }

  private var requestId = 1
  private var ongoingRequest = mutableMapOf<Int, (response: String) -> Unit>()

  private suspend inline fun <reified T : Any?> request(func: String, args: List<Any> = emptyList()): T =
    suspendCancellableCoroutine { continuation ->
      val id = requestId++
      ongoingRequest[id] = { response ->
        logger.info("Agent response: $response")
        val result = gson.fromJson<T>(response, object : TypeToken<T>() {}.type)
        continuation.resumeWith(Result.success(result))
      }
      val data = listOf(id, mapOf("func" to func, "args" to args))
      val json = gson.toJson(data)
      logger.info("Agent request: $json")
      streamWriter.write(json + "\n")
      streamWriter.flush()

      continuation.invokeOnCancellation {
        logger.info("Agent request cancelled")
        val cancellationId = requestId++
        ongoingRequest[cancellationId] = { response ->
          logger.info("Agent cancellation response: $response")
        }
        val cancellationData = listOf(cancellationId, mapOf("func" to "cancelRequest", "args" to listOf(id)))
        val cancellationJson = gson.toJson(cancellationData)
        logger.info("Agent cancellation request: $cancellationJson")
        streamWriter.write(cancellationJson + "\n")
        streamWriter.flush()
      }
    }

  private var outputBuffer: String = ""

  override fun onTextAvailable(event: ProcessEvent, outputType: Key<*>) {
    logger.info("Output received: $outputType: ${event.text}")
    if (outputType !== ProcessOutputTypes.STDOUT) return
    val lines = (outputBuffer + event.text).split("\n")
    lines.subList(0, lines.size - 1).forEach { string -> handleOutput(string) }
    outputBuffer = lines.last()
  }

  private fun handleOutput(output: String) {
    val data = try {
      gson.fromJson(output, Array::class.java).toList()
    } catch (e: Exception) {
      logger.error("Failed to parse agent output: $output")
      return
    }
    if (data.size != 2 || data[0] !is Number) {
      logger.error("Failed to parse agent output: $output")
      return
    }
    logger.info("Parsed agent output: $data")
    val id = (data[0] as Number).toInt()
    if (id == 0) {
      if (data[1] is Map<*, *>) {
        handleNotification(data[1] as Map<*, *>)
      }
    } else {
      ongoingRequest[id]?.let { callback ->
        callback(gson.toJson(data[1]))
      }
      ongoingRequest.remove(id)
    }
  }

  private fun handleNotification(event: Map<*, *>) {
    when (event["event"]) {
      "statusChanged" -> {
        logger.info("Agent notification $event")
        statusFlow.value = when (event["status"]) {
          "notInitialized" -> Status.NOT_INITIALIZED
          "ready" -> Status.READY
          "disconnected" -> Status.DISCONNECTED
          "unauthorized" -> Status.UNAUTHORIZED
          else -> Status.NOT_INITIALIZED
        }
      }
      "configUpdated" -> {
        logger.info("Agent notification $event")
      }
      "authRequired" -> {
        logger.info("Agent notification $event")
      }
      else -> {
        logger.error("Agent notification, unknown event name: ${event["event"]}")
      }
    }
  }
}