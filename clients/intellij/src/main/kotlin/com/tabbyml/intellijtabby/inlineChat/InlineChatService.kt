package com.tabbyml.intellijtabby.inlineChat

import com.intellij.openapi.Disposable
import com.intellij.openapi.components.Service
import com.intellij.openapi.project.Project
import org.eclipse.lsp4j.Location

@Service(Service.Level.PROJECT)
class InlineChatService(private val project: Project) : Disposable {

    var inlineChatEditing = false
    var location: Location? = null

    override fun dispose() {
    }
}