package com.tabbyml.intellijtabby.settings

import com.intellij.openapi.actionSystem.KeyboardShortcut
import com.intellij.openapi.actionSystem.Shortcut
import com.intellij.openapi.components.Service
import com.intellij.openapi.components.service
import com.intellij.openapi.keymap.KeymapManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch

@Service
class KeymapService {
  init {
    val settings = service<ApplicationSettingsState>()
    CoroutineScope(Dispatchers.IO).launch {
      settings.keymapStyleState.collect {
        applyKeymapStyle(it)
      }
    }
  }

  private fun applyKeymapStyle(style: ApplicationSettingsState.KeymapStyle) {
    val manager = KeymapManager.getInstance()
    val keymap = manager.activeKeymap
    when (style) {
      ApplicationSettingsState.KeymapStyle.DEFAULT -> {
        keymap.removeAllActionShortcuts("Tabby.AcceptCompletion")
        keymap.removeAllActionShortcuts("Tabby.AcceptCompletionNextLine")
        keymap.removeAllActionShortcuts("Tabby.AcceptCompletionNextWord")
        keymap.addShortcut("Tabby.AcceptCompletion", KeyboardShortcut.fromString("TAB"))
        keymap.addShortcut("Tabby.AcceptCompletionNextLine", KeyboardShortcut.fromString("ctrl TAB"))
        keymap.addShortcut("Tabby.AcceptCompletionNextWord", KeyboardShortcut.fromString("ctrl RIGHT"))
      }

      ApplicationSettingsState.KeymapStyle.TABBY_STYLE -> {
        keymap.removeAllActionShortcuts("Tabby.AcceptCompletion")
        keymap.removeAllActionShortcuts("Tabby.AcceptCompletionNextLine")
        keymap.removeAllActionShortcuts("Tabby.AcceptCompletionNextWord")
        keymap.addShortcut("Tabby.AcceptCompletion", KeyboardShortcut.fromString("ctrl TAB"))
        keymap.addShortcut("Tabby.AcceptCompletionNextLine", KeyboardShortcut.fromString("TAB"))
        keymap.addShortcut("Tabby.AcceptCompletionNextWord", KeyboardShortcut.fromString("ctrl RIGHT"))
      }

      ApplicationSettingsState.KeymapStyle.CUSTOM -> {
        // Do nothing
      }
    }
  }

  fun getTriggerShortcut(): String? {
    val manager = KeymapManager.getInstance()
    val keymap = manager.activeKeymap
    return keymap.getShortcuts("Tabby.TriggerCompletion").firstOrNull()?.toString()
  }

  private var shortcutStore = mutableMapOf<String, List<Shortcut>>()

  fun onShowInlineCompletion() {
    storeConflicts("Tabby.AcceptCompletion")
    storeConflicts("Tabby.AcceptCompletionNextLine")
    storeConflicts("Tabby.AcceptCompletionNextWord")
  }

  fun onDismissInlineCompletion() {
    val manager = KeymapManager.getInstance()
    val keymap = manager.activeKeymap
    shortcutStore.forEach { (actionId, shortcuts) ->
      shortcuts.forEach { shortcut ->
        keymap.addShortcut(actionId, shortcut)
      }
    }
    shortcutStore.clear()
  }

  private fun storeConflicts(actionId: String) {
    val manager = KeymapManager.getInstance()
    val keymap = manager.activeKeymap
    val shortcuts = keymap.getShortcuts(actionId)
    shortcuts.forEach {
      if (it is KeyboardShortcut) {
        val conflicts = keymap.getConflicts(actionId, it)
        shortcutStore.putAll(conflicts)
        conflicts.forEach { (conflictActionId, conflictShortcuts) ->
          conflictShortcuts.forEach { conflictShortcut ->
            keymap.removeShortcut(conflictActionId, conflictShortcut)
          }
        }
      }
    }
  }
}