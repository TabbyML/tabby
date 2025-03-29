package com.tabbyml.intellijtabby.inlineChat

import com.intellij.codeInsight.hints.*
import com.intellij.lang.Language
import com.intellij.openapi.components.serviceOrNull
import com.intellij.openapi.editor.Editor
import com.intellij.psi.PsiElement
import com.intellij.psi.PsiFile
import com.tabbyml.intellijtabby.lsp.ConnectionService
import kotlinx.coroutines.CoroutineScope
import org.eclipse.lsp4j.CodeLens
import org.eclipse.lsp4j.CodeLensParams
import org.eclipse.lsp4j.TextDocumentIdentifier
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutionException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import javax.swing.JComponent
import javax.swing.JPanel


class CodeLensInlayHintsProvider :
    InlayHintsProvider<Any> {
    private val scope = CoroutineScope(Dispatchers.IO)

    override fun getCollectorFor(
        file: PsiFile,
        editor: Editor,
        settings: Any,
        sink: InlayHintsSink
    ): InlayHintsCollector? {
        return object : FactoryInlayHintsCollector(editor) {
            override fun collect(element: PsiElement, editor: Editor, sink: InlayHintsSink): Boolean {
                try {
                    val codeLenses = getCodeLenses(element.containingFile).get() ?: return false
                    for (codeLens in codeLenses) {
                        val range = codeLens.range
                        if (range == null || codeLens.command == null) continue

                        val start = range.start
                        val line = start.line
                        val offset = editor.document.getLineStartOffset(line)

                        // Create the inlay presentation for the CodeLens
                        val presentation = factory.smallText(codeLens.command.title)

                        // Add to the editor
                        sink.addBlockElement(offset, true, true, 0, presentation)
                    }
                } catch (e: InterruptedException) {
                    // Handle exceptions
                } catch (e: ExecutionException) {
                    // Handle exceptions
                }
                return true
            }
        }
    }

    fun getCodeLenses(psiFile: PsiFile): CompletableFuture<List<CodeLens>?> {
        val virtualFile = psiFile.virtualFile
            ?: return CompletableFuture.completedFuture(listOf())
        val uri = virtualFile.url.replace("file://", "file:/")
        val params = CodeLensParams(TextDocumentIdentifier(uri))
        return CompletableFuture<List<CodeLens>?>().also { future ->
            scope.launch {
                try {
                    val server = psiFile.project.serviceOrNull<ConnectionService>()?.getServerAsync() ?: run {
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

    override fun createSettings(): NoSettings {
        return NoSettings()
    }

    override fun isLanguageSupported(language: Language): Boolean {
        // Specify which languages should support CodeLens
        return true // or check specific languages
    }

    override val key: SettingsKey<Any> = SettingsKey("Tabby")

    override val name: String = "Tabby CodeLens"
    override val previewText: String = "Tabby CodeLens"

    override fun createConfigurable(settings: Any): ImmediateConfigurable {
        return object : ImmediateConfigurable {
            override val cases: List<ImmediateConfigurable.Case> = emptyList()
            override fun createComponent(listener: ChangeListener): JComponent {
                return JPanel()
            }
        }
    }
}
