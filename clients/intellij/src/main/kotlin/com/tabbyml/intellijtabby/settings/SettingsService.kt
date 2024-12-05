package com.tabbyml.intellijtabby.settings

import com.intellij.openapi.components.Service
import com.intellij.openapi.components.SimplePersistentStateComponent
import com.intellij.openapi.components.State
import com.intellij.openapi.components.Storage
import com.intellij.openapi.project.Project
import com.intellij.util.messages.Topic
import com.tabbyml.intellijtabby.safeSyncPublisher

@Service
@State(
  name = "com.tabbyml.intellijtabby.settings.SettingsService", storages = [Storage("intellij-tabby.xml")]
)
class SettingsService : SimplePersistentStateComponent<SettingsState>(SettingsState()) {
  var completionTriggerMode
    get() = state.completionTriggerMode
    set(value) {
      state.completionTriggerMode = value
    }
  var serverEndpoint
    get() = state.serverEndpoint ?: ""
    set(value) {
      state.serverEndpoint = value
    }
  var serverToken
    get() = state.serverToken ?: ""
    set(value) {
      state.serverToken = value
    }
  var nodeBinary
    get() = state.nodeBinary ?: ""
    set(value) {
      state.nodeBinary = value
    }
  var isAnonymousUsageTrackingDisabled
    get() = state.isAnonymousUsageTrackingDisabled
    set(value) {
      state.isAnonymousUsageTrackingDisabled = value
    }

  fun notifyChanges(project: Project) {
    project.safeSyncPublisher(Listener.TOPIC)?.settingsChanged(settings())
  }

  data class Settings(
    val completionTriggerMode: SettingsState.TriggerMode,
    val serverEndpoint: String,
    val serverToken: String,
    val nodeBinary: String,
    val isAnonymousUsageTrackingDisabled: Boolean,
  )

  fun settings(): Settings {
    return Settings(
      completionTriggerMode,
      serverEndpoint,
      serverToken,
      nodeBinary,
      isAnonymousUsageTrackingDisabled,
    )
  }

  interface Listener {
    fun settingsChanged(settings: Settings) {}

    companion object {
      @Topic.ProjectLevel
      val TOPIC = Topic(Listener::class.java, Topic.BroadcastDirection.NONE)
    }
  }
}