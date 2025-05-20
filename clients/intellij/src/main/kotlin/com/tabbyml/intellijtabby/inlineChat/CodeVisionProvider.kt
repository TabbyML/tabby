package com.tabbyml.intellijtabby.inlineChat

import com.google.gson.JsonObject
import com.intellij.codeInsight.codeVision.*
import com.intellij.codeInsight.codeVision.CodeVisionState.Companion.READY_EMPTY
import com.intellij.codeInsight.codeVision.settings.CodeVisionGroupSettingProvider
import com.intellij.codeInsight.codeVision.ui.model.TextCodeVisionEntry
import com.intellij.icons.AllIcons
import com.intellij.ide.DataManager
import com.intellij.openapi.actionSystem.ActionManager
import com.intellij.openapi.actionSystem.ex.ActionUtil
import com.intellij.openapi.components.serviceOrNull
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.fileEditor.FileDocumentManager
import com.intellij.openapi.keymap.KeymapUtil
import com.intellij.openapi.project.DumbAware
import com.intellij.openapi.util.TextRange
import com.intellij.ui.SpinningProgressIcon
import org.eclipse.lsp4j.Location
import org.eclipse.lsp4j.Position
import org.eclipse.lsp4j.Range
import javax.swing.Icon

abstract class InlineChatCodeVisionProvider : CodeVisionProvider<Any>, DumbAware {
    private val logger = Logger.getInstance(InlineChatCodeVisionProvider::class.java)
    override val defaultAnchor: CodeVisionAnchorKind = CodeVisionAnchorKind.Top

    // provider id
    abstract override val id: String

    override val groupId = "Tabby.InlineEdit"

    // command name
    abstract val command: String

    // action name
    abstract val action: String?

    // execute action id
    abstract val actionId: String?
    abstract val icon: Icon

    override fun precomputeOnUiThread(editor: Editor): Any {
        return Any()
    }

    override fun computeCodeVision(editor: Editor, uiData: Any): CodeVisionState {
        val project = editor.project ?: return READY_EMPTY
        val inlineChatService = project.serviceOrNull<InlineChatService>() ?: return READY_EMPTY
        val document = editor.document
        val virtualFile = FileDocumentManager.getInstance()
            .getFile(editor.document)
        val uri = virtualFile?.url ?: return READY_EMPTY
        val codeLenses = try {
            getCodeLenses(project, uri).get() ?: return READY_EMPTY
        } catch (e: Exception) {
            logger.warn("Failed to get code lenses", e)
            return READY_EMPTY
        }
        val codelens = codeLenses.firstOrNull() {
            val codeLensCommand = it.command?.command
            val codeLensAction = (it.command?.arguments?.firstOrNull() as JsonObject?)?.get("action")?.asString
            codeLensCommand == command && codeLensAction == action
        }
        if (codelens == null) {
            inlineChatService.inlineChatDiffActionState[id] = false
            return READY_EMPTY
        }
        inlineChatService.inlineChatDiffActionState[id] = true
        inlineChatService.location = Location(
            uri,
            Range(
                Position(codelens.range.start.line, codelens.range.start.character),
                Position(codelens.range.end.line, codelens.range.end.character)
            )
        )
        val prefixRegex = Regex("""^\$\(.*?\)""")
        val title = codelens.command.title.replace(
            prefixRegex,
            ""
        ) + if (actionId != null) " (${KeymapUtil.getFirstKeyboardShortcutText(getAction(actionId!!))})" else ""
        val startOffset = document.getLineStartOffset(codelens.range.start.line) + codelens.range.start.character
        val endOffset = document.getLineStartOffset(codelens.range.end.line) + codelens.range.end.character
        val entry =
            TextCodeVisionEntry(title, id, icon)
        val textRange = TextRange(startOffset, endOffset)
        textRange to entry
        // CodeVisionProvider can only display one entry for each line
        return CodeVisionState.Ready(listOf(textRange to entry))
    }

    override fun handleClick(editor: Editor, textRange: TextRange, entry: CodeVisionEntry) {
        if (actionId == null) {
            return
        }
        val editorDataContext = DataManager.getInstance().getDataContext(editor.component)
        ActionUtil.invokeAction(getAction(actionId!!), editorDataContext, "", null, null)
    }

    private fun getAction(actionId: String) = ActionManager.getInstance().getAction(actionId)
}

class InlineChatLoadingCodeVisionProvider : InlineChatCodeVisionProvider() {
    override val id: String = "Tabby.InlineChat.Loading"
    override val name ="Tabby Inline Edit Loading"
    override val command: String = " "
    override val action: String? = null
    override val actionId: String? = null
    override val icon: Icon = SpinningProgressIcon()
    override val relativeOrderings: List<CodeVisionRelativeOrdering> =
        listOf(CodeVisionRelativeOrdering.CodeVisionRelativeOrderingBefore("Tabby.InlineChat.Cancel"))
}

class InlineChatCancelCodeVisionProvider : InlineChatCodeVisionProvider() {
    override val id: String = "Tabby.InlineChat.Cancel"
    override val name ="Tabby Inline Edit Cancel"
    override val command: String = "tabby/chat/edit/resolve"
    override val action: String = "cancel"
    override val actionId: String = "Tabby.InlineChat.Resolve.Cancel"
    override val icon: Icon = AllIcons.Actions.Cancel
    override val relativeOrderings: List<CodeVisionRelativeOrdering> =
        emptyList()
}

class InlineChatAcceptCodeVisionProvider : InlineChatCodeVisionProvider() {
    override val id: String = "Tabby.InlineChat.Accept"
    override val name ="Tabby Inline Edit Accept"
    override val command: String = "tabby/chat/edit/resolve"
    override val action: String? = "accept"
    override val actionId: String = "Tabby.InlineChat.Resolve.Accept"
    override val icon: Icon = AllIcons.Actions.Checked
    override val relativeOrderings: List<CodeVisionRelativeOrdering> =
        listOf(CodeVisionRelativeOrdering.CodeVisionRelativeOrderingBefore("Tabby.InlineChat.Discard"))
}

class InlineChatDiscardCodeVisionProvider : InlineChatCodeVisionProvider() {
    override val id: String = "Tabby.InlineChat.Discard"
    override val name ="Tabby Inline Edit Discard"
    override val command: String = "tabby/chat/edit/resolve"
    override val action: String = "discard"
    override val actionId: String = "Tabby.InlineChat.Resolve.Discard"
    override val icon: Icon = AllIcons.Actions.Close
    override val relativeOrderings: List<CodeVisionRelativeOrdering> =
        emptyList()
}

class InlineChatCodeVisionSettingProvider: CodeVisionGroupSettingProvider {
    override val groupId: String
        get() = "Tabby.InlineEdit"
    override val groupName: String
        get() = "Tabby Inline Edit"
    override val description: String
        get() = "Tabby Inline Edit"
}