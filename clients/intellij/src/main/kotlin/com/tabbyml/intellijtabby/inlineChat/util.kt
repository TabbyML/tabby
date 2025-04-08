package com.tabbyml.intellijtabby.inlineChat

import com.intellij.openapi.components.serviceOrNull
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.project.Project
import com.tabbyml.intellijtabby.lsp.ConnectionService
import com.tabbyml.intellijtabby.lsp.protocol.ChatEditCommand
import com.tabbyml.intellijtabby.lsp.protocol.ChatEditCommandParams
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.eclipse.lsp4j.*
import java.util.concurrent.CompletableFuture

data class LocationInfo(val location: Location, val startOffset: Int)

fun getCurrentLocation(editor: Editor): LocationInfo {
    val fileUri = editor.virtualFile.url
    val location = Location()
    location.uri = fileUri
    val selectionModel = editor.selectionModel
    val document = editor.document
    val caretOffset = editor.caretModel.offset
    var startOffset = caretOffset
    var endOffset = caretOffset
    if (selectionModel.hasSelection()) {
        startOffset = selectionModel.selectionStart
        endOffset = selectionModel.selectionEnd
    }
    val startPosition = Position(document.getLineNumber(startOffset), 0)
    val endChar = endOffset - document.getLineStartOffset(document.getLineNumber(endOffset))
    val endLine = if (endChar == 0) document.getLineNumber(endOffset) else document.getLineNumber(endOffset) + 1
    val endPosition = Position(endLine, 0)
    location.range = Range(startPosition, endPosition)
    return LocationInfo(location, startOffset)
}

fun getCodeLenses(project: Project, uri: String): CompletableFuture<List<CodeLens>?> {
    val scope = CoroutineScope(Dispatchers.IO)
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

fun getSuggestedCommands(project: Project, location: Location): CompletableFuture<List<ChatEditCommand>?> {
    val scope = CoroutineScope(Dispatchers.IO)
    val params = ChatEditCommandParams(location)
    return CompletableFuture<List<ChatEditCommand>?>().also { future ->
        scope.launch {
            try {
                val server = project.serviceOrNull<ConnectionService>()?.getServerAsync() ?: run {
                    future.complete(null)
                    return@launch
                }
                val result = server.chatFeature.editCommand(params)
                future.complete(result.get())
            } catch (e: Exception) {
                future.completeExceptionally(e)
            }
        }
    }
}