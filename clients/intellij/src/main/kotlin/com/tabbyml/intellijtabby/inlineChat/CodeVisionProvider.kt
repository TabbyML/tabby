package com.tabbyml.intellijtabby.inlineChat

import com.intellij.codeInsight.codeVision.*
import com.intellij.openapi.editor.Editor
import javax.swing.Icon


class InlineChatCodeVisionProvider: CodeVisionProvider<Any> {
    override val defaultAnchor: CodeVisionAnchorKind = CodeVisionAnchorKind.Top
    override val id: String = "InlineChatCodeVisionProvider"
    override val name: String = "Inline Chat Code Vision Provider"
    override val relativeOrderings: List<CodeVisionRelativeOrdering> = listOf()

    override fun precomputeOnUiThread(editor: Editor): Any {
        return Any()
    }

    override fun computeCodeVision(editor: Editor, uiData: Any): CodeVisionState {
        TODO()
    }

}