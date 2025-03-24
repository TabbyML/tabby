package com.tabbyml.intellijtabby.inlineChat

import com.intellij.codeInsight.daemon.LineMarkerInfo
import com.intellij.codeInsight.daemon.LineMarkerProvider
import com.intellij.openapi.editor.markup.GutterIconRenderer
import com.intellij.psi.PsiElement
import com.intellij.psi.util.PsiEditorUtil
import com.tabbyml.intellijtabby.Icons

class GutterIconProvider: LineMarkerProvider {
    override fun getLineMarkerInfo(element: PsiElement): LineMarkerInfo<*> {
        return LineMarkerInfo(
            element,
            element.textRange,
            Icons.Chat,
            { "Click to open tabby inline chat" },
            { _, elt ->
                val project = elt.project
                val editor = PsiEditorUtil.findEditor(elt)
                if (editor != null) {
                    InlineChatIntentionAction().invoke(project, editor, elt.containingFile)
                }
            },
            GutterIconRenderer.Alignment.CENTER,
            { "tabby inline chat gutter icon" }
        )
    }
}