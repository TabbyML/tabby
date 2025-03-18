package com.tabbyml.intellijtabby.events

import com.intellij.openapi.Disposable
import com.intellij.openapi.components.Service
import com.intellij.openapi.project.Project
import com.intellij.util.messages.Topic
import com.tabbyml.intellijtabby.lsp.LanguageClient
import com.tabbyml.intellijtabby.safeSyncPublisher

@Service(Service.Level.PROJECT)
class FeaturesState(private val project: Project) : Disposable {
  private val messageBusConnection = project.messageBus.connect()
  private val registrations = mutableMapOf<String, Pair<String, Any>>()

  data class Features(
    val inlineCompletion: Boolean,
    val chat: Boolean,
  )

  val features: Features
    get() = Features(
      inlineCompletion = registrations.containsKey("textDocument/inlineCompletion"),
      chat = registrations.containsKey("tabby/chat"),
    )

  init {
    messageBusConnection.subscribe(
      LanguageClient.CapabilityRegistrationListener.TOPIC,
      object : LanguageClient.CapabilityRegistrationListener {
        override fun onRegisterCapability(id: String, method: String, options: Any) {
          registrations[method] = Pair(id, options)
          project.safeSyncPublisher(Listener.TOPIC)?.stateChanged(features)
        }

        override fun onUnregisterCapability(id: String, method: String) {
          registrations.remove(method)
          project.safeSyncPublisher(Listener.TOPIC)?.stateChanged(features)
        }
      })
  }

  override fun dispose() {
    messageBusConnection.dispose()
  }

  interface Listener {
    fun stateChanged(features: Features) {}

    companion object {
      @Topic.ProjectLevel
      val TOPIC = Topic(Listener::class.java, Topic.BroadcastDirection.NONE)
    }
  }
}