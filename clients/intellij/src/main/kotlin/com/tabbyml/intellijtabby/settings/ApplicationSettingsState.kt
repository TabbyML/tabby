package com.tabbyml.intellijtabby.settings

import com.google.gson.annotations.SerializedName
import com.intellij.openapi.components.PersistentStateComponent
import com.intellij.openapi.components.Service
import com.intellij.openapi.components.State
import com.intellij.openapi.components.Storage
import com.intellij.util.xmlb.XmlSerializerUtil
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

@Service
@State(
  name = "com.tabbyml.intellijtabby.settings.ApplicationSettingsState",
  storages = [Storage("intellij-tabby.xml")]
)
class ApplicationSettingsState : PersistentStateComponent<ApplicationSettingsState> {
  enum class TriggerMode {
    @SerializedName("manual")
    MANUAL,
    @SerializedName("automatic")
    AUTOMATIC,
  }

  var completionTriggerMode: TriggerMode = TriggerMode.AUTOMATIC
    set(value) {
      field = value
      stateFlow.value = this.data
    }
  var serverEndpoint: String = ""
    set(value) {
      field = value
      stateFlow.value = this.data
    }
  var isAnonymousUsageTrackingDisabled: Boolean = false
    set(value) {
      field = value
      stateFlow.value = this.data
    }

  data class State(
    val completionTriggerMode: TriggerMode,
    val serverEndpoint: String,
    val isAnonymousUsageTrackingDisabled: Boolean,
  )

  val data: State
    get() = State(
      completionTriggerMode = completionTriggerMode,
      serverEndpoint = serverEndpoint,
      isAnonymousUsageTrackingDisabled = isAnonymousUsageTrackingDisabled,
    )

  private val stateFlow = MutableStateFlow(data)
  val state = stateFlow.asStateFlow()

  override fun getState(): ApplicationSettingsState {
    return this
  }

  override fun loadState(state: ApplicationSettingsState) {
    XmlSerializerUtil.copyBean(state, this)
  }
}