package com.tabbyml.intellijtabby

import com.intellij.openapi.util.IconLoader
import com.intellij.ui.NewUI
import javax.swing.Icon

private fun loadIcon(name: String): Icon {
  return if (NewUI.isEnabled()) {
    IconLoader.getIcon("/icons/new-ui/$name", Icons::class.java)
  } else {
    IconLoader.getIcon("/icons/$name", Icons::class.java)
  }
}

object Icons {
  @JvmField
  val Chat = loadIcon("chat.svg")
}