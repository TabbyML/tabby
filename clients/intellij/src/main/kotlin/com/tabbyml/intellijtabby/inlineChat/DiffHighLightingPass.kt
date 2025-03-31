package com.tabbyml.intellijtabby.inlineChat

import com.google.gson.JsonObject
import com.intellij.codeHighlighting.*
import com.intellij.openapi.editor.colors.EditorColors
import com.intellij.codeInsight.daemon.impl.HighlightInfo
import com.intellij.codeInsight.daemon.impl.HighlightInfoType
import com.intellij.codeInsight.daemon.impl.UpdateHighlightersUtil
import com.intellij.lang.annotation.HighlightSeverity
import com.intellij.openapi.application.invokeLater
import com.intellij.openapi.components.serviceOrNull
import com.intellij.openapi.editor.Document
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.colors.EditorColorsManager
import com.intellij.openapi.editor.markup.RangeHighlighter
import com.intellij.openapi.editor.markup.TextAttributes
import com.intellij.openapi.fileEditor.FileDocumentManager
import com.intellij.openapi.progress.ProgressIndicator
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
import java.awt.Font
import java.util.concurrent.CompletableFuture

class DiffHighlighterRegister : TextEditorHighlightingPassFactoryRegistrar {
    override fun registerHighlightingPassFactory(register: TextEditorHighlightingPassRegistrar, project: Project) {
        register.registerTextEditorHighlightingPass(
            DiffHighlightingPassFactory(), TextEditorHighlightingPassRegistrar.Anchor.LAST,
            Pass.UPDATE_ALL, false, false
        )
    }
}

class DiffHighlightingPassFactory : TextEditorHighlightingPassFactory {
    override fun createHighlightingPass(file: PsiFile, editor: Editor): TextEditorHighlightingPass? {
        // You can add conditions to determine if the pass should be created
        if (!file.isValid || !shouldCreateHighlightingPass(file, editor)) {
            return null
        }

        return DiffHighLightingPass(file.project, editor.document, editor)
    }

    private fun shouldCreateHighlightingPass(file: PsiFile, editor: Editor): Boolean {
        // Your logic to determine if the pass should be created
        // For example, check file type
        return true
    }
}

class DiffHighLightingPass(project: Project, document: Document, val editor: Editor) :
    TextEditorHighlightingPass(project, document, true) {

    private var lenses = emptyList<CodeLens>()
    private val file = FileDocumentManager.getInstance().getFile(myDocument)
    private val myHighlights = mutableListOf<HighlightInfo>()

    override fun doCollectInformation(progress: ProgressIndicator) {
        colorsScheme = EditorColorsManager.getInstance().globalScheme
        val headerColor = Color(64f/255, 166f/255, 1f, 0.5f)
        val insertColor = Color((155f / 255), 185f / 255, 85f / 255, 0.2f)
        val lineAttribuitesMap = mapOf<String, TextAttributes>(
            "header" to TextAttributes(
                null,
                headerColor,
                null,
                null,
                0
            ),
            "footer" to TextAttributes(
                null,
                headerColor,
                null,
                null,
                0
            ),
            "commentsFirstLine" to TextAttributes(
                null,
                colorsScheme?.getColor(EditorColors.DOCUMENTATION_COLOR),
                null,
                null,
                Font.ITALIC
            ),
            "comments" to TextAttributes(
                null,
                colorsScheme?.getColor(EditorColors.DOCUMENTATION_COLOR),
                null,
                null,
                Font.ITALIC
            ),
            "waiting" to TextAttributes(
                null,
                colorsScheme?.getColor(EditorColors.TAB_UNDERLINE_INACTIVE),
                null,
                null,
                0
            ),
            "inProgress" to TextAttributes(null, insertColor, null, null, 0),
            "unchanged" to TextAttributes(null, null, null, null, 0),
            "inserted" to TextAttributes(null, insertColor, null, null, 0),
            "deleted" to TextAttributes(null, Color(1f, 0f, 0f, 0.2f), null, null, 0),
        )

        val textAttributesMap = mapOf<String, TextAttributes>(
            "inserted" to TextAttributes(null, Color(156f / 255, 204f / 255, 44f / 255, 0.2f), null, null, 0),
            "deleted" to TextAttributes(
                null,
                Color(1f, 0f, 0f, 0.2f),
                null,
                null,
                0
            ),
        )
        val uri = file?.url ?: return
        lenses = getCodeLenses(myProject, uri).get() ?: emptyList()
        for (lens in lenses) {
            if ((lens.data as JsonObject?)?.get("type")?.asString != "previewChanges") continue
            val range = lens.range
            val startOffset = myDocument.getLineStartOffset(range.start.line) + range.start.character
            val lineType = (lens.data as JsonObject?)?.get("line")?.asString
            var endOffset = myDocument.getLineStartOffset(range.end.line) + range.end.character
            if (lineType != null) {
                endOffset = myDocument.getLineStartOffset(range.end.line + 1)
            }
            val textRange = TextRange(startOffset, endOffset)
            var attributes = TextAttributes(null, null, null, null, 0)
            if (lineType != null) {
                attributes = lineAttribuitesMap.get(lineType) ?: continue
            }
            val textType = (lens.data as JsonObject?)?.get("text")?.asString
            if (textType != null) {
                attributes = textAttributesMap.get(textType) ?: continue
            }
            val builder = HighlightInfo.newHighlightInfo(HighlightInfoType.INFORMATION)
                .range(textRange)
                .textAttributes(attributes)
                .descriptionAndTooltip("Tabby inline diff")
                .severity(HighlightSeverity.TEXT_ATTRIBUTES)
            val highlight = builder.create() ?: continue
            myHighlights.add(highlight)
        }
    }

    override fun doApplyInformationToEditor() {
        UpdateHighlightersUtil.setHighlightersToEditor(
            myProject, myDocument, 0, myDocument.textLength,
            myHighlights, colorsScheme, id
        )
    }

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
}