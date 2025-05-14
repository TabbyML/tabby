package com.tabbyml.intellijtabby.lsp

import com.intellij.execution.configurations.GeneralCommandLine
import com.intellij.execution.configurations.PathEnvironmentVariableUtil
import com.intellij.ide.plugins.PluginManagerCore
import com.intellij.openapi.Disposable
import com.intellij.openapi.components.Service
import com.intellij.openapi.components.service
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.extensions.PluginId
import com.intellij.openapi.project.Project
import com.intellij.util.EnvironmentUtil
import com.intellij.util.messages.Topic
import com.tabbyml.intellijtabby.lsp.protocol.server.LanguageServer
import com.tabbyml.intellijtabby.notifications.notifyInitializationFailed
import com.tabbyml.intellijtabby.safeSyncPublisher
import com.tabbyml.intellijtabby.settings.SettingsService
import kotlinx.coroutines.delay
import kotlinx.coroutines.future.await
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.eclipse.lsp4j.InitializedParams
import org.eclipse.lsp4j.jsonrpc.Launcher
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader
import java.io.PrintWriter
import java.util.concurrent.Future
import java.util.concurrent.TimeUnit

@Service(Service.Level.PROJECT)
class ConnectionService(private val project: Project) : Disposable {
  private val logger = Logger.getInstance(ConnectionService::class.java)
  private val settings = service<SettingsService>()
  private val client = LanguageClient(project)
  private var process: Process? = null
  private var listening: Future<Void>? = null
  private var server: LanguageServer? = null
  private val initializeMutex = Mutex()

  suspend fun getServerAsync(): LanguageServer? {
    if (server == null || listening == null || (process?.isAlive != true)) {
      initializeMutex.withLock {
        if (server == null || listening == null || (process?.isAlive != true)) {
          initialize()
        }
      }
    }
    return server
  }

  open class InitializationException(message: String) : Exception(message)

  open class NodeBinaryException(message: String) : InitializationException(
    message = "$message Please install Node.js version >= 18.0, set the binary path in Tabby plugin settings or add bin path to system environment variable PATH, then restart IDE."
  )

  open class NodeBinaryNotFoundException : NodeBinaryException(
    message = "Cannot find Node binary."
  )

  open class NodeBinaryInvalidVersionException(version: String) : NodeBinaryException(
    message = "Node version is too old: $version."
  )

  private suspend fun initialize(retry: Int = 0) {
    try {
      logger.info("Creating tabby-agent process...")
      project.safeSyncPublisher(Listener.TOPIC)?.connectionStateChanged(State.INITIALIZING)
      val node = getNodeBinary()
      val script = getNodeScript()
      val options = "--stdio"
      val cmd = GeneralCommandLine(node.absolutePath, script.absolutePath, options).withCharset(Charsets.UTF_8)
      val process = cmd.createProcess()
      if (!process.isAlive) {
        throw InitializationException("Failed to create agent process.")
      }
      val launcher =
        Launcher.Builder<LanguageServer>().setLocalService(client).setRemoteInterface(LanguageServer::class.java)
          .setInput(process.inputStream).setOutput(process.outputStream).traceMessages(PrintWriter(Tracer())).create()
      val server = launcher.remoteProxy
      logger.info("Created tabby-agent process with PID: ${process.pid()}, listening to stdio.")

      this.process = process
      this.server = server
      this.listening = launcher.startListening()

      val initializeParams = client.buildInitializeParams()
      val initializeResult = server.initialize(initializeParams).await()
      client.processInitializeResult(server, initializeResult)
      server.initialized(InitializedParams())
      project.safeSyncPublisher(Listener.TOPIC)?.connectionStateChanged(State.READY)
    } catch (e: InitializationException) {
      logger.warn("Failed to initialize connection.", e)
      if (retry < 5) {
        val initRetryDelay: Long = 1000
        delay(initRetryDelay)
        initialize(retry + 1)
      } else {
        project.safeSyncPublisher(Listener.TOPIC)?.connectionStateChanged(State.INITIALIZATION_FAILED)
        notifyInitializationFailed(e)
      }
    }
  }

  private suspend fun shutdown() {
    try {
      server?.let { server ->
        server.shutdown().orTimeout(3, TimeUnit.SECONDS).await()
        server.exit()
        this.server = null

        listening?.let {
          it.cancel(true)
          this.listening = null
        }

        process?.let {
          if (it.isAlive) {
            it.destroy()
          }
          this.process = null
        }
      }
    } catch (e: Exception) {
      logger.warn("Failed to shutdown.", e)
    }
  }

  private fun getNodeBinary(): File {
    val node = settings.nodeBinary.let {
      if (it.isNotBlank()) {
        val path = it.replaceFirst(Regex("^~"), System.getProperty("user.home"))
        File(path)
      } else {
        logger.info("Environment variables: PATH: ${EnvironmentUtil.getValue("PATH")}")
        PathEnvironmentVariableUtil.findExecutableInPathOnAnyOS("node")
      }
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
      if (e is InitializationException) {
        throw e
      } else {
        throw InitializationException("Failed to check node version: $e.")
      }
    }
  }

  private fun getNodeScript(): File {
    val script =
      PluginManagerCore.getPlugin(PluginId.getId("com.tabbyml.intellij-tabby"))?.pluginPath?.resolve("tabby-agent/node/index.js")
        ?.toFile()
    if (script?.exists() == true) {
      logger.info("Node script path: ${script.absolutePath}")
      return script
    } else {
      throw InitializationException("Node script not found. Please reinstall Tabby plugin.")
    }
  }

  override fun dispose() {
    runBlocking {
      shutdown()
      client.dispose()
    }
  }

  enum class State {
    INITIALIZING, READY, INITIALIZATION_FAILED,
  }

  interface Listener {
    fun connectionStateChanged(state: State) {}

    companion object {
      @Topic.ProjectLevel
      val TOPIC = Topic(Listener::class.java, Topic.BroadcastDirection.NONE)
    }
  }
}