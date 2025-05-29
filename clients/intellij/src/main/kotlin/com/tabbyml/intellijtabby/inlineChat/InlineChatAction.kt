package com.tabbyml.intellijtabby.inlineChat

import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.actionSystem.CommonDataKeys
import com.intellij.openapi.components.serviceOrNull
import com.intellij.openapi.project.DumbAwareAction
import com.tabbyml.intellijtabby.actions.chat.isChatFeatureEnabled
import com.tabbyml.intellijtabby.lsp.ConnectionService
import com.tabbyml.intellijtabby.lsp.protocol.ChatEditResolveParams
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

abstract class BaseInlineChatAction : DumbAwareAction() {
    override fun update(e: AnActionEvent) {
        val project = e.project
        e.presentation.isEnabled = isChatFeatureEnabled(project)
    }
}

class InlineChatAction : BaseInlineChatAction() {
    override fun actionPerformed(e: AnActionEvent) {
        val editor = e.getRequiredData(CommonDataKeys.EDITOR)
        val project = e.project ?: return
        InlineChatIntentionAction().invoke(project, editor, null)
    }
}

class InlineChatAcceptAction : BaseInlineChatAction() {
    private val scope = CoroutineScope(Dispatchers.IO)

    override fun actionPerformed(e: AnActionEvent) {
        val project = e.project ?: return
        val inlineChatService = project.serviceOrNull<InlineChatService>() ?: return
        val location = inlineChatService.location ?: return
        scope.launch {
            val server = project.serviceOrNull<ConnectionService>()?.getServerAsync() ?: return@launch
            server.chatFeature.resolveEdit(ChatEditResolveParams(location = location, action = "accept"))
        }
    }
}

class InlineChatDiscardAction : BaseInlineChatAction() {
    private val scope = CoroutineScope(Dispatchers.IO)

    override fun actionPerformed(e: AnActionEvent) {
        val project = e.project ?: return
        val inlineChatService = project.serviceOrNull<InlineChatService>() ?: return
        val location = inlineChatService.location ?: return
        scope.launch {
            val server = project.serviceOrNull<ConnectionService>()?.getServerAsync() ?: return@launch
            server.chatFeature.resolveEdit(ChatEditResolveParams(location = location, action = "discard"))
        }
    }
}

class InlineChatCancelAction : BaseInlineChatAction() {
    private val scope = CoroutineScope(Dispatchers.IO)

    override fun actionPerformed(e: AnActionEvent) {
        val project = e.project ?: return
        val inlineChatService = project.serviceOrNull<InlineChatService>() ?: return
        val location = inlineChatService.location ?: return
        scope.launch {
            val server = project.serviceOrNull<ConnectionService>()?.getServerAsync() ?: return@launch
            server.chatFeature.resolveEdit(ChatEditResolveParams(location = location, action = "cancel"))
        }
    }
}