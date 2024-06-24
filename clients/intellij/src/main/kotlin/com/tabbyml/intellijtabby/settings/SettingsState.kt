package com.tabbyml.intellijtabby.settings

import com.intellij.openapi.components.PersistentStateComponent
import com.intellij.openapi.components.Service
import com.intellij.openapi.components.State
import com.intellij.openapi.components.Storage
import com.intellij.openapi.project.Project
import com.intellij.util.messages.Topic
import com.intellij.util.xmlb.XmlSerializerUtil

@Service(Service.Level.PROJECT)
@State(
  name = "com.tabbyml.intellijtabby.settings.SettingsState", storages = [Storage("intellij-tabby.xml")]
)
class SettingsState(private val project: Project) : PersistentStateComponent<SettingsState> {
  private val publisher = project.messageBus.syncPublisher(Listener.TOPIC)

  enum class TriggerMode {
    MANUAL, AUTOMATIC,
  }

  var completionTriggerMode: TriggerMode = TriggerMode.AUTOMATIC
  var serverEndpoint: String = ""
  var serverToken: String = ""
  var nodeBinary: String = ""
  var isAnonymousUsageTrackingDisabled: Boolean = false
  var notificationsMuted: List<String> = listOf()

  fun notifyChanges() {
    publisher.settingsChanged(settings())
  }

  data class Settings(
    val completionTriggerMode: TriggerMode,
    val serverEndpoint: String,
    val serverToken: String,
    val nodeBinary: String,
    val isAnonymousUsageTrackingDisabled: Boolean,
    val notificationsMuted: List<String>,
  )

  fun settings(): Settings {
    return Settings(
      completionTriggerMode,
      serverEndpoint,
      serverToken,
      nodeBinary,
      isAnonymousUsageTrackingDisabled,
      notificationsMuted,
    )
  }

  override fun getState(): SettingsState {
    return this
  }

  override fun loadState(state: SettingsState) {
    XmlSerializerUtil.copyBean(state, this)
  }

  interface Listener {
    fun settingsChanged(settings: Settings) {}

    companion object {
      @Topic.ProjectLevel
      val TOPIC = Topic(Listener::class.java, Topic.BroadcastDirection.NONE)
    }
  }
}