package com.tabbyml.intellijtabby.inlineChat

import com.intellij.lang.annotation.AnnotationHolder
import com.intellij.lang.annotation.ExternalAnnotator
import com.intellij.openapi.components.serviceOrNull
import com.intellij.openapi.editor.colors.TextAttributesKey
import com.intellij.openapi.editor.markup.TextAttributes
import com.intellij.openapi.fileEditor.FileDocumentManager
import com.intellij.openapi.project.Project
import com.intellij.openapi.util.TextRange
import com.intellij.psi.PsiFile
import com.tabbyml.intellijtabby.lsp.ConnectionService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.eclipse.lsp4j.CodeLens
import org.eclipse.lsp4j.CodeLensParams
import org.eclipse.lsp4j.TextDocumentIdentifier
import java.awt.Color
import java.util.concurrent.CompletableFuture


interface InitInfo {
    val project: Project;
    val uri: String;
}

class DiffHighlighter: ExternalAnnotator<InitInfo, List<CodeLens>>() {
    private val scope = CoroutineScope(Dispatchers.IO)

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

    override fun collectInformation(file: PsiFile): InitInfo? {

        val project = file.project
        val uri = file.virtualFile?.url ?: return null
        return object : InitInfo {
            override val project: Project = project
            override val uri: String = uri
        }
    }

    override fun doAnnotate(collectedInfo: InitInfo?): List<CodeLens>? {
        val project = collectedInfo?.project ?: return null
        val uri = collectedInfo.uri
        return getCodeLenses(project, uri).get()
    }
    
    override fun apply(file: PsiFile, annotationResult: List<CodeLens>?, holder: AnnotationHolder) {
        if (annotationResult == null) return
        val document = FileDocumentManager.getInstance().getDocument(file.virtualFile!!) ?: return
        for (codeLens in annotationResult) {
            val range = codeLens.range
            val startOffset = document.getLineStartOffset(range.start.line) + range.start.character
            val endOffset = document.getLineStartOffset(range.end.line) + range.end.character
            val textRange = TextRange(startOffset, endOffset)
            // Add to the editor
            val attributes = TextAttributes()
            attributes.backgroundColor = Color(200, 255, 200)
            holder.newAnnotation(
                com.intellij.lang.annotation.HighlightSeverity.INFORMATION,
                codeLens.command.title
            )
                .range(textRange)
                .textAttributes(TextAttributesKey.createTempTextAttributesKey("DiffHighlighter", attributes))
                .create()
        }
    }

}