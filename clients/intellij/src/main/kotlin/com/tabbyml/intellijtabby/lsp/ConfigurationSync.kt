package com.tabbyml.intellijtabby.lsp

import com.intellij.openapi.Disposable
import com.intellij.openapi.components.service
import com.intellij.openapi.keymap.Keymap
import com.intellij.openapi.keymap.KeymapManagerListener
import com.intellij.openapi.project.Project
import com.tabbyml.intellijtabby.lsp.protocol.ClientProvidedConfig
import com.tabbyml.intellijtabby.lsp.protocol.DidChangeConfigurationParams
import com.tabbyml.intellijtabby.lsp.protocol.server.LanguageServer
import com.tabbyml.intellijtabby.settings.KeymapSettings
import com.tabbyml.intellijtabby.settings.SettingsState

class ConfigurationSync(private val project: Project) : Disposable {
  private val messageBusConnection = project.messageBus.connect()
  private val settings = project.service<SettingsState>()
  private val keymapSettings = project.service<KeymapSettings>()

  data class SettingsData(
    val settings: SettingsState.Settings,
    val keymap: KeymapSettings.KeymapStyle,
  ) {
    fun withSettings(settings: SettingsState.Settings): SettingsData {
      return SettingsData(settings, keymap)
    }

    fun withKeymap(keymap: KeymapSettings.KeymapStyle): SettingsData {
      return SettingsData(settings, keymap)
    }
  }

  private var cached: SettingsData = SettingsData(
    settings.settings(),
    keymapSettings.getCurrentKeymapStyle(),
  )

  fun getConfiguration(): ClientProvidedConfig {
    cached = SettingsData(
      settings.settings(),
      keymapSettings.getCurrentKeymapStyle(),
    )
    return buildClientProvidedConfig(cached)
  }

  fun startSync(server: LanguageServer) {
    messageBusConnection.subscribe(SettingsState.Listener.TOPIC, object : SettingsState.Listener {
      override fun settingsChanged(settings: SettingsState.Settings) {
        cached = cached.withSettings(settings)
        notifyServer(server)
      }
    })
    messageBusConnection.subscribe(KeymapManagerListener.TOPIC, object : KeymapManagerListener {
      override fun shortcutsChanged(keymap: Keymap, actionIds: MutableCollection<String>, fromSettings: Boolean) {
        val current = keymapSettings.getCurrentKeymapStyle()
        if (cached.keymap !== current) {
          cached = cached.withKeymap(current)
          notifyServer(server)
        }
      }
    })
  }

  private fun notifyServer(server: LanguageServer) {
    server.workspaceFeature.didChangeConfiguration(
      DidChangeConfigurationParams(
        settings = buildClientProvidedConfig(cached)
      )
    )
  }

  private fun buildClientProvidedConfig(data: SettingsData): ClientProvidedConfig {
    val settings = data.settings
    val keymap = data.keymap
    return ClientProvidedConfig(
      server = if (settings.serverEndpoint.isNotBlank() || settings.serverToken.isNotBlank()) {
        ClientProvidedConfig.ServerConfig(
          endpoint = settings.serverEndpoint.ifBlank { null },
          token = settings.serverToken.ifBlank { null },
        )
      } else {
        null
      },
      inlineCompletion = ClientProvidedConfig.InlineCompletionConfig(
        triggerMode = when (settings.completionTriggerMode) {
          SettingsState.TriggerMode.AUTOMATIC -> ClientProvidedConfig.InlineCompletionConfig.TriggerMode.AUTO
          SettingsState.TriggerMode.MANUAL -> ClientProvidedConfig.InlineCompletionConfig.TriggerMode.MANUAL
        }
      ),
      keybindings = when (keymap) {
        KeymapSettings.KeymapStyle.DEFAULT -> ClientProvidedConfig.KeybindingsConfig.DEFAULT
        KeymapSettings.KeymapStyle.TABBY_STYLE -> ClientProvidedConfig.KeybindingsConfig.TABBY_STYLE
        KeymapSettings.KeymapStyle.CUSTOMIZE -> ClientProvidedConfig.KeybindingsConfig.CUSTOMIZE
      },
      anonymousUsageTracking = if (settings.isAnonymousUsageTrackingDisabled) {
        ClientProvidedConfig.AnonymousUsageTrackingConfig(
          disable = true,
        )
      } else {
        null
      },
    )
  }

  override fun dispose() {
    messageBusConnection.dispose()
  }
}