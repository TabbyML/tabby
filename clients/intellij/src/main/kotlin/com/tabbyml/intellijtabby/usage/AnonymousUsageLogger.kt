package com.tabbyml.intellijtabby.usage

import com.google.gson.Gson
import com.intellij.openapi.components.Service
import com.intellij.openapi.components.service
import com.intellij.openapi.diagnostic.Logger
import com.tabbyml.intellijtabby.settings.ApplicationSettingsState
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import java.util.*
import kotlin.io.path.Path

@Service
class AnonymousUsageLogger {
  private val logger = Logger.getInstance(AnonymousUsageLogger::class.java)
  private val gson = Gson()
  private val scope: CoroutineScope = CoroutineScope(Dispatchers.IO)
  private lateinit var anonymousId: String
  private val disabled: Boolean
    get() {
      return service<ApplicationSettingsState>().isAnonymousUsageTrackingDisabled
    }
  private val initialized = MutableStateFlow(false)

  init {
    scope.launch {
      try {
        val home = System.getProperty("user.home")
        logger.info("User home: $home")
        val datafile = Path(home).resolve(".tabby/agent/data.json").toFile()
        var data: Map<*, *>? = null
        try {
          val dataJson = datafile.inputStream().bufferedReader().use { it.readText() }
          data = gson.fromJson(dataJson, Map::class.java)
        } catch (e: Exception) {
          logger.error("Failed to load anonymous ID: ${e.message}")
        }
        if (data?.get("anonymousId") != null) {
          anonymousId = data["anonymousId"].toString()
          logger.info("Saved anonymous ID: $anonymousId")
        } else {
          anonymousId = UUID.randomUUID().toString()
          val newData = data?.toMutableMap() ?: mutableMapOf()
          newData["anonymousId"] = anonymousId
          val newDataJson = gson.toJson(newData)
          datafile.writeText(newDataJson)
          logger.info("Create new anonymous ID: $anonymousId")
        }
      } catch (e: Exception) {
        logger.error("Failed when init anonymous ID: ${e.message}")
        anonymousId = UUID.randomUUID().toString()
      } finally {
        initialized.value = true
      }
    }
  }

  data class UsageRequest(
    val distinctId: String,
    val event: String,
    val properties: Map<String, String>,
  )

  suspend fun event(event: String, properties: Map<String, String>) {
    initialized.first { it }

    if (disabled) {
      return
    }

    val request = UsageRequest(
      distinctId = anonymousId,
      event = event,
      properties = properties,
    )
    val requestString = gson.toJson(request)

    withContext(scope.coroutineContext) {
      try {
        val connection = URL(ENDPOINT).openConnection() as HttpURLConnection
        connection.requestMethod = "POST"
        connection.setRequestProperty("Content-Type", "application/json")
        connection.setRequestProperty("Accept", "application/json")
        connection.doInput = true
        connection.doOutput = true

        val outputStreamWriter = OutputStreamWriter(connection.outputStream)
        outputStreamWriter.write(requestString)
        outputStreamWriter.flush()

        val responseCode = connection.responseCode
        if (responseCode == HttpURLConnection.HTTP_OK) {
          logger.info("Usage event sent successfully.")
        } else {
          logger.error("Usage event failed to send: $responseCode")
        }
        connection.disconnect()
      } catch (e: Exception) {
        logger.error("Usage event failed to send: ${e.message}")
      }
    }

  }

  companion object {
    const val ENDPOINT = "https://app.tabbyml.com/api/usage"
  }
}