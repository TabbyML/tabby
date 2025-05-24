package com.tabbyml.intellijtabby.inlineChat

import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.application.invokeLater
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.event.SelectionEvent
import com.intellij.openapi.editor.event.SelectionListener
import com.intellij.openapi.editor.markup.GutterIconRenderer
import com.intellij.openapi.editor.markup.HighlighterLayer
import com.intellij.openapi.editor.markup.HighlighterTargetArea
import com.intellij.openapi.editor.markup.RangeHighlighter
import com.intellij.openapi.fileEditor.FileEditorManager
import com.intellij.openapi.fileEditor.FileEditorManagerEvent
import com.intellij.openapi.fileEditor.FileEditorManagerListener
import com.intellij.openapi.fileEditor.TextEditor
import com.intellij.openapi.project.Project
import com.intellij.openapi.startup.ProjectActivity
import com.intellij.openapi.util.IconLoader
import com.intellij.openapi.vfs.VirtualFile
import com.tabbyml.intellijtabby.actions.chat.isChatFeatureEnabled
import javax.swing.Icon

class SelectionGutterIconManager : ProjectActivity {
    private val selectionIcon: Icon by lazy {
        IconLoader.getIcon("/icons/chat.svg", SelectionGutterIconManager::class.java)
    }

    private val editorToHighlighter = mutableMapOf<Editor, RangeHighlighter>()
    private var activeEditor: Editor? = null
    private var currentSelectionListener: SelectionListener? = null

    override suspend fun execute(project: Project) {
        project.messageBus.connect().subscribe(
            FileEditorManagerListener.FILE_EDITOR_MANAGER,
            object : FileEditorManagerListener {
                override fun fileOpened(source: FileEditorManager, file: VirtualFile) {
                    val editor = source.selectedTextEditor
                    editor?.let { setActiveEditor(it) }
                }

                override fun fileClosed(source: FileEditorManager, file: VirtualFile) {
                    val editors = source.getEditors(file)
                    for (fileEditor in editors) {
                        if (fileEditor is TextEditor) {
                            val editor = fileEditor.editor
                            if (editor == activeEditor) {
                                removeSelectionListener(editor)
                                removeHighlighter(editor)
                                activeEditor = null
                            }
                        }
                    }
                }

                override fun selectionChanged(event: FileEditorManagerEvent) {
                    if (event.newEditor != null) {
                        val editor = FileEditorManager.getInstance(project).selectedTextEditor
                        if (editor != null && editor != activeEditor) {
                            activeEditor?.let {
                                removeSelectionListener(it)
                                removeHighlighter(it)
                            }
                            setActiveEditor(editor)
                        }
                    }
                }
            }
        )

        FileEditorManager.getInstance(project).selectedTextEditor?.let {
            setActiveEditor(it)
        }
    }

    private fun setActiveEditor(editor: Editor) {
        activeEditor = editor
        addSelectionListener(editor)
        updateSelectionGutterIcon(editor)
    }

    private fun addSelectionListener(editor: Editor) {
        currentSelectionListener = object : SelectionListener {
            override fun selectionChanged(e: SelectionEvent) {
                updateSelectionGutterIcon(editor)
            }
        }
        editor.selectionModel.addSelectionListener(currentSelectionListener!!)
    }

    private fun removeSelectionListener(editor: Editor) {
        currentSelectionListener?.let {
            editor.selectionModel.removeSelectionListener(it)
            currentSelectionListener = null
        }
    }

    private fun updateSelectionGutterIcon(editor: Editor) {
        if (!isChatFeatureEnabled(editor.project)) return
        invokeLater {
            removeHighlighter(editor)

            if (!editor.selectionModel.hasSelection()) return@invokeLater

            val selectionStart = editor.selectionModel.selectionStart
            val lineNumber = editor.document.getLineNumber(selectionStart)
            val lineStart = editor.document.getLineStartOffset(lineNumber)

            val markupModel = editor.markupModel
            val highlighter = markupModel.addRangeHighlighter(
                lineStart, lineStart + 1,
                HighlighterLayer.LAST,
                null,
                HighlighterTargetArea.LINES_IN_RANGE
            )

            highlighter.gutterIconRenderer = object : GutterIconRenderer() {
                override fun getIcon(): Icon = selectionIcon

                override fun equals(other: Any?): Boolean {
                    return other is GutterIconRenderer && other.javaClass == this.javaClass
                }

                override fun hashCode(): Int = javaClass.hashCode()

                override fun getTooltipText(): String = "Open Tabby inline edit"

                override fun getClickAction(): AnAction {
                    return InlineChatAction()
                }
            }

            editorToHighlighter[editor] = highlighter
        }
    }

    private fun removeHighlighter(editor: Editor) {
        editorToHighlighter.remove(editor)?.let { highlighter ->
            if (highlighter.isValid) {
                editor.markupModel.removeHighlighter(highlighter)
            }
        }
    }
}