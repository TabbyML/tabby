package com.tabbyml.intellijtabby.settings

import com.intellij.openapi.components.BaseState


class SettingsState : BaseState() {
  enum class TriggerMode {
    MANUAL, AUTOMATIC,
  }

  var completionTriggerMode by enum(TriggerMode.AUTOMATIC)
  var serverEndpoint by string()
  var serverToken by string()
  var nodeBinary by string()
  var isAnonymousUsageTrackingDisabled by property(false)
}