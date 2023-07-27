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
import java.io.OutputStreamWriter
import java.util.concurrent.CompletableFuture

class Agent : ProcessAdapter() {
  private val logger = Logger.getInstance(Agent::class.java)
  private val gson = Gson()
  private val process: KillableProcessHandler
  private val streamWriter: OutputStreamWriter

  var status = "notInitialized"
    private set

  init {
    logger.info("Agent init.")
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
    process = object: KillableProcessHandler(cmd) {
      override fun readerOptions(): BaseOutputReader.Options {
        return BaseOutputReader.Options.forMostlySilentProcess()
      }
    }
    process.startNotify()
    process.addProcessListener(this)
    streamWriter = process.processInput.writer()
  }

  fun initialize(): CompletableFuture<Boolean> {
    return request("initialize", listOf(mapOf("client" to "intellij-tabby")))
  }

  fun updateConfig(): CompletableFuture<Boolean> {
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

  fun getCompletions(request: CompletionRequest): CompletableFuture<CompletionResponse?> {
    return request("getCompletions", listOf(request))
  }

  private var requestId = 1
  private var ongoingRequest = mutableMapOf<Int, (response: String) -> Unit>()

  private inline fun <reified T : Any?> request(func: String, args: List<Any> = emptyList()): CompletableFuture<T> {
    val id = requestId++
    val data = listOf(id, mapOf("func" to func, "args" to args))
    val json = gson.toJson(data)
    streamWriter.write(json + "\n")
    streamWriter.flush()
    logger.info("Agent request: $json")
    val future = CompletableFuture<T>()
    ongoingRequest[id] = { response ->
      logger.info("Agent response: $response")
      val result = gson.fromJson<T>(response, object : TypeToken<T>() {}.type)
      future.complete(result)
    }
    return future
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
      handleNotification(gson.toJson(data[1]))
    } else {
      ongoingRequest[id]?.let { callback ->
        callback(gson.toJson(data[1]))
      }
      ongoingRequest.remove(id)
    }
  }

  private fun handleNotification(event: String) {
    logger.info("Agent notification: $event")
  }
}