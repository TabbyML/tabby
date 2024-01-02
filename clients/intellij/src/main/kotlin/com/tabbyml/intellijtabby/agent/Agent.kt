package com.tabbyml.intellijtabby.agent

import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import com.google.gson.reflect.TypeToken
import com.intellij.execution.configurations.GeneralCommandLine
import com.intellij.execution.configurations.PathEnvironmentVariableUtil
import com.intellij.execution.process.KillableProcessHandler
import com.intellij.execution.process.ProcessAdapter
import com.intellij.execution.process.ProcessEvent
import com.intellij.execution.process.ProcessOutputTypes
import com.intellij.ide.plugins.PluginManagerCore
import com.intellij.openapi.components.service
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.extensions.PluginId
import com.intellij.openapi.util.Key
import com.intellij.util.EnvironmentUtil
import com.intellij.util.io.BaseOutputReader
import com.tabbyml.intellijtabby.settings.ApplicationSettingsState
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.suspendCancellableCoroutine
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader
import java.io.OutputStreamWriter

class Agent : ProcessAdapter() {
  private val logger = Logger.getInstance(Agent::class.java)
  private val gson = Gson()
  private lateinit var process: KillableProcessHandler
  private lateinit var streamWriter: OutputStreamWriter

  enum class Status {
    NOT_INITIALIZED,
    READY,
    DISCONNECTED,
    UNAUTHORIZED,
  }

  private val statusFlow = MutableStateFlow(Status.NOT_INITIALIZED)
  val status = statusFlow.asStateFlow()
  private val authRequiredEventFlow = MutableSharedFlow<Unit>(extraBufferCapacity = 1)
  val authRequiredEvent = authRequiredEventFlow.asSharedFlow()
  private val currentIssueFlow = MutableStateFlow<String?>(null)
  val currentIssue = currentIssueFlow.asStateFlow()

  open class AgentException(message: String) : Exception(message)

  open class NodeBinaryException(message: String) : AgentException(
    message = "$message Please install Node.js version >= 18.0, set the binary path in Tabby plugin settings or add bin path to system environment variable PATH, then restart IDE."
  )

  open class NodeBinaryNotFoundException : NodeBinaryException(
    message = "Cannot find Node binary."
  )

  open class NodeBinaryInvalidVersionException(version: String) : NodeBinaryException(
    message = "Node version is too old: $version."
  )

  fun open() {
    val node = getNodeBinary()
    val script = getNodeScript()
    val options = "--dns-result-order=ipv4first"
    val cmd = GeneralCommandLine(node.absolutePath, options, script.absolutePath).withCharset(Charsets.UTF_8)
    process = object : KillableProcessHandler(cmd) {
      override fun readerOptions(): BaseOutputReader.Options {
        return BaseOutputReader.Options.forMostlySilentProcess()
      }
    }
    process.startNotify()
    process.addProcessListener(this)
    streamWriter = process.processInput.writer()
  }

  private fun getNodeBinary(): File {
    val settings = service<ApplicationSettingsState>()
    val node = if (settings.nodeBinary.isNotBlank()) {
      val path = settings.nodeBinary.replaceFirst(Regex("^~"), System.getProperty("user.home"))
      File(path)
    } else {
      logger.info("Environment variables: PATH: ${EnvironmentUtil.getValue("PATH")}")
      PathEnvironmentVariableUtil.findExecutableInPathOnAnyOS("node")
    }

    if (node?.exists() == true) {
      logger.info("Node binary path: ${node.absolutePath}")
      checkNodeVersion(node)
      return node
    } else {
      throw NodeBinaryNotFoundException()
    }
  }

  private fun checkNodeVersion(node: File) {
    try {
      val process = GeneralCommandLine(node.absolutePath, "--version").createProcess()
      val version = BufferedReader(InputStreamReader(process.inputStream)).readLine()
      val regResult = Regex("v([0-9]+)\\.([0-9]+)\\.([0-9]+)").find(version)
      if (regResult != null && regResult.groupValues[1].toInt() >= 18) {
        return
      } else {
        throw NodeBinaryInvalidVersionException(version)
      }
    } catch (e: Exception) {
      if (e is AgentException) {
        throw e
      } else {
        throw AgentException("Failed to check node version: $e.")
      }
    }
  }

  private fun getNodeScript(): File {
    val script =
      PluginManagerCore.getPlugin(PluginId.getId("com.tabbyml.intellij-tabby"))?.pluginPath?.resolve("node_scripts/tabby-agent.js")
        ?.toFile()
    if (script?.exists() == true) {
      logger.info("Node script path: ${script.absolutePath}")
      return script
    } else {
      throw AgentException("Node script not found. Please reinstall Tabby plugin.")
    }
  }

  data class Config(
    val server: Server? = null,
    val logs: Logs? = null,
    val anonymousUsageTracking: AnonymousUsageTracking? = null,
  ) {
    data class Server(
      val endpoint: String? = null,
      val token: String? = null,
      val requestHeaders: Map<String, String>? = null,
    )

    data class Logs(
      val level: String? = null,
    )

    data class AnonymousUsageTracking(
      val disabled: Boolean? = null,
    )
  }

  data class ClientProperties(
    val user: Map<String, Any>,
    val session: Map<String, Any>,
  )

  suspend fun initialize(config: Config, clientProperties: ClientProperties): Boolean {
    return request(
      "initialize", listOf(
        mapOf(
          "config" to config,
          "clientProperties" to clientProperties,
        )
      )
    )
  }

  suspend fun finalize(): Boolean {
    return request("finalize", listOf())
  }

  suspend fun updateClientProperties(type: String, key: String, value: Any): Boolean {
    return request("updateClientProperties", listOf(type, key, value))
  }

  suspend fun updateConfig(key: String, config: Any): Boolean {
    return request("updateConfig", listOf(key, config))
  }

  suspend fun clearConfig(key: String): Boolean {
    return request("clearConfig", listOf(key))
  }

  suspend fun getConfig(): Config {
    return request("getConfig", listOf())
  }

  suspend fun getStatus(): Status {
    return request("getStatus", listOf())
  }

  suspend fun getIssues(): List<String> {
    return request("getIssues", listOf())
  }

  data class GetIssueDetailOptions(
    val index: Int? = null,
    val name: String? = null,
  )

  suspend fun getIssueDetail(options: GetIssueDetailOptions): Map<String, Any>? {
    return request("getIssueDetail", listOf(options))
  }

  suspend fun getServerHealthState(): Map<String, Any>? {
    return request("getServerHealthState", listOf())
  }

  data class AuthUrlResponse(
    val authUrl: String,
    val code: String,
  )

  @Deprecated("Tabby Cloud auth support will be removed.")
  suspend fun requestAuthUrl(): AuthUrlResponse? {
    return request("requestAuthUrl", listOf(ABORT_SIGNAL_ENABLED))
  }

  @Deprecated("Tabby Cloud auth support will be removed.")
  suspend fun waitForAuthToken(code: String) {
    return request("waitForAuthToken", listOf(code, ABORT_SIGNAL_ENABLED))
  }

  data class CompletionRequest(
    val filepath: String,
    val language: String,
    val text: String,
    val position: Int,
    val manually: Boolean?,
  )

  data class CompletionResponse(
    val id: String,
    val choices: List<Choice>,
  ) {
    data class Choice(
      val index: Int,
      val text: String,
      val replaceRange: Range,
    ) {
      data class Range(
        val start: Int,
        val end: Int,
      )
    }
  }

  suspend fun provideCompletions(request: CompletionRequest): CompletionResponse? {
    return request("provideCompletions", listOf(request, ABORT_SIGNAL_ENABLED))
  }

  data class LogEventRequest(
    val type: EventType,
    @SerializedName("completion_id") val completionId: String,
    @SerializedName("choice_index") val choiceIndex: Int,
    @SerializedName("select_kind") val selectKind: SelectKind? = null,
  ) {
    enum class EventType {
      @SerializedName("view")
      VIEW,

      @SerializedName("select")
      SELECT,
    }

    enum class SelectKind {
      @SerializedName("line")
      LINE,
    }
  }

  suspend fun postEvent(event: LogEventRequest) {
    request<Any>("postEvent", listOf(event, ABORT_SIGNAL_ENABLED))
  }


  fun close() {
    try {
      streamWriter.close()
      process.killProcess()
    } catch (e: Exception) {
      // ignore
    }
  }

  private var requestId = 1
  private var ongoingRequest = mutableMapOf<Int, (response: String) -> Unit>()

  private suspend inline fun <reified T : Any?> request(func: String, args: List<Any> = emptyList()): T =
    suspendCancellableCoroutine { continuation ->
      val id = requestId++
      ongoingRequest[id] = { response ->
        logger.debug("Agent response: $response")
        val result = gson.fromJson<T>(response, object : TypeToken<T>() {}.type)
        continuation.resumeWith(Result.success(result))
      }
      val data = listOf(id, mapOf("func" to func, "args" to args))
      val json = gson.toJson(data)
      logger.debug("Agent request: $json")
      streamWriter.write(json + "\n")
      streamWriter.flush()

      continuation.invokeOnCancellation {
        logger.debug("Agent request cancelled")
        val cancellationId = requestId++
        ongoingRequest[cancellationId] = { response ->
          logger.debug("Agent cancellation response: $response")
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
    logger.debug("Output received: $outputType: ${event.text}")
    if (outputType !== ProcessOutputTypes.STDOUT) return
    val lines = (outputBuffer + event.text).lines()
    lines.subList(0, lines.size - 1).forEach { string -> handleOutput(string) }
    outputBuffer = lines.last()
  }

  private fun handleOutput(output: String) {
    val data = try {
      gson.fromJson(output, Array::class.java).toList()
    } catch (e: Exception) {
      logger.warn("Failed to parse agent output: $output")
      return
    }
    if (data.size != 2 || data[0] !is Number) {
      logger.warn("Failed to parse agent output: $output")
      return
    }
    logger.debug("Parsed agent output: $data")
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
        logger.debug("Agent notification $event")
        statusFlow.value = when (event["status"]) {
          "notInitialized" -> Status.NOT_INITIALIZED
          "ready" -> Status.READY
          "disconnected" -> Status.DISCONNECTED
          "unauthorized" -> Status.UNAUTHORIZED
          else -> Status.NOT_INITIALIZED
        }
      }

      "configUpdated" -> {
        logger.debug("Agent notification $event")
      }

      "authRequired" -> {
        logger.debug("Agent notification $event")
        authRequiredEventFlow.tryEmit(Unit)
      }

      "issuesUpdated" -> {
        logger.debug("Agent notification $event")
        currentIssueFlow.value = (event["issues"] as List<*>).firstOrNull() as String?
      }

      else -> {
        logger.warn("Agent notification, unknown event name: ${event["event"]}")
      }
    }
  }

  companion object {
    private val ABORT_SIGNAL_ENABLED = mapOf("signal" to true)
  }
}