package com.tabbyml.intellijtabby.inlineChat

import com.intellij.codeInsight.daemon.LineMarkerInfo
import com.intellij.codeInsight.daemon.LineMarkerProvider
import com.intellij.icons.AllIcons
import com.intellij.openapi.editor.markup.GutterIconRenderer
import com.intellij.openapi.fileEditor.FileEditorManager
import com.intellij.psi.PsiElement
import com.tabbyml.intellijtabby.Icons

class GutterIconProvider: LineMarkerProvider {
    override fun getLineMarkerInfo(element: PsiElement): LineMarkerInfo<PsiElement>? {
        return null
    }

    override fun collectSlowLineMarkers(
        elements: MutableList<out PsiElement>,
        result: MutableCollection<in LineMarkerInfo<*>>
    ) {
        elements.forEach {
            val project = it.project
            val editor = FileEditorManager.getInstance(project).selectedTextEditor

            if (editor == null || !editor.selectionModel.hasSelection()) {
                return
            }

            val selectionStartOffset: Int = editor.selectionModel.selectionStart
            val selectionStartLine: Int = editor.document.getLineNumber(selectionStartOffset)
            val elementLine: Int = editor.document.getLineNumber(it.textOffset)

            if (elementLine == selectionStartLine && result.isEmpty()) {
                val markInfo = LineMarkerInfo(it, it.textRange, AllIcons.Actions.Edit, {"Open Tabby Inline Chat"}, null, GutterIconRenderer.Alignment.CENTER) { "Open Tabby Inline Chat" }
                result.add(markInfo)
            }
        }
    }
}