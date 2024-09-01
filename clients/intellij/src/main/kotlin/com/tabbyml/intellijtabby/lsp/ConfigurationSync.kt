package com.tabbyml.intellijtabby.lsp

import com.intellij.openapi.Disposable
import com.intellij.openapi.components.service
import com.intellij.openapi.components.serviceOrNull
import com.intellij.openapi.keymap.Keymap
import com.intellij.openapi.keymap.KeymapManagerListener
import com.intellij.openapi.project.Project
import com.tabbyml.intellijtabby.lsp.protocol.ClientProvidedConfig
import com.tabbyml.intellijtabby.lsp.protocol.DidChangeConfigurationParams
import com.tabbyml.intellijtabby.lsp.protocol.server.LanguageServer
import com.tabbyml.intellijtabby.settings.KeymapSettings
import com.tabbyml.intellijtabby.settings.SettingsService
import com.tabbyml.intellijtabby.settings.SettingsState

class ConfigurationSync(private val project: Project) : Disposable {
  private val messageBusConnection = project.messageBus.connect()
  private val settings = service<SettingsService>()
  private val keymapSettings = project.serviceOrNull<KeymapSettings>()

  data class SettingsData(
    val settings: SettingsService.Settings,
    val keymap: KeymapSettings.KeymapStyle?,
  ) {
    fun withSettings(settings: SettingsService.Settings): SettingsData {
      return SettingsData(settings, keymap)
    }

    fun withKeymap(keymap: KeymapSettings.KeymapStyle): SettingsData {
      return SettingsData(settings, keymap)
    }
  }

  private var cached: SettingsData = SettingsData(
    settings.settings(),
    keymapSettings?.getCurrentKeymapStyle(),
  )

  fun getConfiguration(): ClientProvidedConfig {
    cached = SettingsData(
      settings.settings(),
      keymapSettings?.getCurrentKeymapStyle(),
    )
    return buildClientProvidedConfig(cached)
  }

  fun startSync(server: LanguageServer) {
    messageBusConnection.subscribe(SettingsService.Listener.TOPIC, object : SettingsService.Listener {
      override fun settingsChanged(settings: SettingsService.Settings) {
        cached = cached.withSettings(settings)
        notifyServer(server)
      }
    })
    messageBusConnection.subscribe(KeymapManagerListener.TOPIC, object : KeymapManagerListener {
      override fun shortcutChanged(keymap: Keymap, actionId: String, fromSettings: Boolean) {
        keymapSettings?.getCurrentKeymapStyle()?.let {
          if (cached.keymap !== it) {
            cached = cached.withKeymap(it)
            notifyServer(server)
          }
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
      server = ClientProvidedConfig.ServerConfig(
        endpoint = settings.serverEndpoint,
        token = settings.serverToken,
      ),
      inlineCompletion = ClientProvidedConfig.InlineCompletionConfig(
        triggerMode = when (settings.completionTriggerMode) {
          SettingsState.TriggerMode.AUTOMATIC -> ClientProvidedConfig.InlineCompletionConfig.TriggerMode.AUTO
          SettingsState.TriggerMode.MANUAL -> ClientProvidedConfig.InlineCompletionConfig.TriggerMode.MANUAL
        }
      ),
      keybindings = when (keymap) {
        KeymapSettings.KeymapStyle.DEFAULT -> ClientProvidedConfig.Keybindings.DEFAULT
        KeymapSettings.KeymapStyle.TABBY_STYLE -> ClientProvidedConfig.Keybindings.TABBY_STYLE
        KeymapSettings.KeymapStyle.CUSTOMIZE -> ClientProvidedConfig.Keybindings.CUSTOMIZE
        null -> ClientProvidedConfig.Keybindings.DEFAULT
      },
      anonymousUsageTracking = ClientProvidedConfig.AnonymousUsageTrackingConfig(
        disable = settings.isAnonymousUsageTrackingDisabled,
      ),
    )
  }

  override fun dispose() {
    messageBusConnection.dispose()
  }
}