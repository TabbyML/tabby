package com.tabbyml.intellijtabby

import com.intellij.openapi.project.Project
import com.intellij.util.messages.Topic

fun <L : Any> Project.safeSyncPublisher(topic: Topic<L>): L? {
  return if (isDisposed) {
    null
  } else {
    messageBus.let {
      if (it.isDisposed) {
        null
      } else {
        it.syncPublisher(topic)
      }
    }
  }
}
