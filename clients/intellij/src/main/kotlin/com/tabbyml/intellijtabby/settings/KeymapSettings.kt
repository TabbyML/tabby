package com.tabbyml.intellijtabby.settings

import com.intellij.openapi.actionSystem.KeyboardShortcut
import com.intellij.openapi.components.Service
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.keymap.Keymap
import com.intellij.openapi.keymap.ex.KeymapManagerEx
import com.intellij.openapi.project.Project

@Service(Service.Level.PROJECT)
class KeymapSettings(private val project: Project) {
  private val logger = Logger.getInstance(KeymapSettings::class.java)
  private val manager = KeymapManagerEx.getInstanceEx()

  enum class KeymapStyle {
    DEFAULT, TABBY_STYLE, CUSTOMIZE,
  }

  fun getCurrentKeymapStyle(): KeymapStyle {
    val keymap = manager.activeKeymap
    val style = if (isSchemeMatched(keymap, DEFAULT_KEYMAP_SCHEMA)) {
      KeymapStyle.DEFAULT
    } else if (isSchemeMatched(keymap, TABBY_STYLE_KEYMAP_SCHEMA)) {
      KeymapStyle.TABBY_STYLE
    } else {
      KeymapStyle.CUSTOMIZE
    }
    logger.info("Current keymap style: $style ($keymap)")
    return style
  }

  private fun isSchemeMatched(keymap: Keymap, schema: Map<String, List<KeyboardShortcut>>): Boolean {
    for ((actionId, shortcuts) in schema) {
      val actionShortcuts = keymap.getShortcuts(actionId)
      if (actionShortcuts.size != shortcuts.size) {
        return false
      }
      for (shortcut in shortcuts) {
        if (!actionShortcuts.any { it.equals(shortcut) }) {
          return false
        }
      }
    }
    return true
  }

  fun applyKeymapStyle(style: KeymapStyle) {
    logger.info("Apply keymap style: $style")
    val keymap = manager.activeKeymap
    when (style) {
      KeymapStyle.DEFAULT -> {
        for ((actionId, shortcuts) in DEFAULT_KEYMAP_SCHEMA) {
          keymap.removeAllActionShortcuts(actionId)
          for (shortcut in shortcuts) {
            keymap.addShortcut(actionId, shortcut)
          }
        }
      }

      KeymapStyle.TABBY_STYLE -> {
        for ((actionId, shortcuts) in TABBY_STYLE_KEYMAP_SCHEMA) {
          keymap.removeAllActionShortcuts(actionId)
          for (shortcut in shortcuts) {
            keymap.addShortcut(actionId, shortcut)
          }
        }
      }

      KeymapStyle.CUSTOMIZE -> {
        // Do nothing
      }
    }
  }

  companion object {
    private val DEFAULT_KEYMAP_SCHEMA = mapOf(
      "Tabby.InlineCompletion.Trigger" to listOf(
        KeyboardShortcut.fromString("ctrl BACK_SLASH"), KeyboardShortcut.fromString("alt BACK_SLASH")
      ),
      "Tabby.InlineCompletion.TabAccept" to listOf(KeyboardShortcut.fromString("TAB")),
      "Tabby.InlineCompletion.AcceptNextLine" to listOf(KeyboardShortcut.fromString("ctrl TAB")),
      "Tabby.InlineCompletion.AcceptNextWord" to listOf(KeyboardShortcut.fromString("ctrl RIGHT")),
      "Tabby.InlineCompletion.Dismiss" to listOf(KeyboardShortcut.fromString("ESCAPE")),
      "Tabby.Chat.ToggleChatToolWindow" to listOf(KeyboardShortcut.fromString("ctrl L")),
      "Tabby.InlineChat.Open" to listOf(KeyboardShortcut.fromString("ctrl I")),
      "Tabby.InlineChat.Resolve.Accept" to listOf(KeyboardShortcut.fromString("ctrl shift D")),
      "Tabby.InlineChat.Resolve.Discard" to listOf(KeyboardShortcut.fromString("ctrl ESCAPE")),
    )
    private val TABBY_STYLE_KEYMAP_SCHEMA = mapOf(
      "Tabby.InlineCompletion.Trigger" to listOf(
        KeyboardShortcut.fromString("ctrl BACK_SLASH"), KeyboardShortcut.fromString("alt BACK_SLASH")
      ),
      "Tabby.InlineCompletion.TabAccept" to listOf(KeyboardShortcut.fromString("ctrl TAB")),
      "Tabby.InlineCompletion.AcceptNextLine" to listOf(KeyboardShortcut.fromString("TAB")),
      "Tabby.InlineCompletion.AcceptNextWord" to listOf(KeyboardShortcut.fromString("ctrl RIGHT")),
      "Tabby.InlineCompletion.Dismiss" to listOf(KeyboardShortcut.fromString("ESCAPE")),
      "Tabby.Chat.ToggleChatToolWindow" to listOf(KeyboardShortcut.fromString("ctrl L")),
      "Tabby.InlineChat.Open" to listOf(KeyboardShortcut.fromString("ctrl I")),
      "Tabby.InlineChat.Resolve.Accept" to listOf(KeyboardShortcut.fromString("ctrl shift D")),
      "Tabby.InlineChat.Resolve.Discard" to listOf(KeyboardShortcut.fromString("ctrl ESCAPE")),
    )
  }
}