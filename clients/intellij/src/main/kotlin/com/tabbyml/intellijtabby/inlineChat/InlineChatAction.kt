package com.tabbyml.intellijtabby.inlineChat

import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.actionSystem.CommonDataKeys
import com.intellij.psi.util.PsiEditorUtil
import org.jetbrains.kotlin.idea.refactoring.psiElement

class InlineChatAction: AnAction() {
    override fun actionPerformed(e: AnActionEvent) {
        val editor = e.getRequiredData(CommonDataKeys.EDITOR)
        val project = e.project ?: return
        InlineChatIntentionAction().invoke(project, editor, null)
    }
}