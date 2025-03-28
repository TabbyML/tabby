package com.tabbyml.intellijtabby.inlineChat

import com.intellij.codeInsight.codeVision.*
import com.intellij.codeInsight.codeVision.CodeVisionState.Companion.READY_EMPTY
import com.intellij.codeInsight.codeVision.ui.model.ClickableTextCodeVisionEntry
import com.intellij.icons.AllIcons
import com.intellij.openapi.components.serviceOrNull
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.fileEditor.FileDocumentManager
import com.intellij.openapi.project.Project
import com.intellij.openapi.util.TextRange
import com.intellij.psi.PsiDocumentManager
import com.tabbyml.intellijtabby.lsp.ConnectionService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.eclipse.lsp4j.CodeLens
import org.eclipse.lsp4j.CodeLensParams
import org.eclipse.lsp4j.TextDocumentIdentifier
import java.util.concurrent.CompletableFuture


class InlineChatCodeVisionProvider : CodeVisionProvider<Any> {
    private val logger = Logger.getInstance(InlineChatCodeVisionProvider::class.java)

    private val scope = CoroutineScope(Dispatchers.IO)
    override val defaultAnchor: CodeVisionAnchorKind = CodeVisionAnchorKind.Top
    override val id: String = "InlineChatCodeVisionProvider"
    override val name: String = "Inline Chat Code Vision Provider"
    override val relativeOrderings: List<CodeVisionRelativeOrdering> = listOf(CodeVisionRelativeOrdering.CodeVisionRelativeOrderingFirst)

    override fun precomputeOnUiThread(editor: Editor): Any {
        return Any()
    }

    override fun computeCodeVision(editor: Editor, uiData: Any): CodeVisionState {
        val project = editor.project ?: return READY_EMPTY
        val document = editor.document
        val virtualFile = FileDocumentManager.getInstance()
            .getFile(editor.document)
        val uri = virtualFile?.url ?: return READY_EMPTY
        val codeLenses = getCodeLenses(project, uri).get() ?: return READY_EMPTY
        println("Code lenses: $codeLenses")
//                && (it.command?.arguments?.firstOrNull() as? Map<*, *>)?.get("action") != null
        val lenses: List<Pair<TextRange, CodeVisionEntry>> = codeLenses.filter { it.command != null  }.mapIndexed { index, it ->
            val startOffset = document.getLineStartOffset(it.range.start.line) + it.range.start.character
            val endOffset = document.getLineStartOffset(it.range.end.line) + it.range.end.character
            val title = it.command.title
            val entry =
                ClickableTextCodeVisionEntry(title, id, { _, _ -> }, AllIcons.Actions.Close, "", "", emptyList())
            val textRange = TextRange(startOffset + index, startOffset + index + 1)
            textRange to entry
        }
        println("lenses: $lenses")
        return CodeVisionState.Ready(lenses)
    }

    private fun getCodeLenses(project: Project, uri: String): CompletableFuture<List<CodeLens>?> {
        val params = CodeLensParams(TextDocumentIdentifier(uri))
        return CompletableFuture<List<CodeLens>?>().also { future ->
            scope.launch {
                try {
                    val server = project.serviceOrNull<ConnectionService>()?.getServerAsync() ?: run {
                        future.complete(null)
                        return@launch
                    }
                    val result = server.textDocumentFeature.codeLens(params)
                    future.complete(result.get())
                } catch (e: Exception) {
                    future.completeExceptionally(e)
                }
            }
        }
    }

}