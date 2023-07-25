package com.tabbyml.intellijtabby.editor

import com.intellij.openapi.application.ReadAction
import com.intellij.openapi.command.WriteCommandAction
import com.intellij.openapi.components.Service
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.EditorCustomElementRenderer
import com.intellij.openapi.editor.Inlay
import com.intellij.openapi.editor.markup.TextAttributes
import com.intellij.openapi.util.Disposer
import com.intellij.ui.JBColor
import com.tabbyml.intellijtabby.agent.Agent
import java.awt.Graphics
import java.awt.Rectangle

@Service
class InlineCompletionService {
  private val logger = Logger.getInstance(InlineCompletionService::class.java)
  var currentText: String? = null
    private set
  var currentOffset: Int? = null
    private set
  private var currentInlays: MutableList<Inlay<*>> = mutableListOf()

  fun show(editor: Editor, offset: Int, completion: Agent.CompletionResponse) {
    if (completion.choices.isEmpty()) {
      return
    }
    val text = completion.choices.first().text
    logger.info("Showing inline completion at $offset: $text")
    val lines = text.split("\n")
    lines.forEachIndexed { index, line -> addInlayLine(editor, offset, line, index) }
    currentText = text
    currentOffset = offset
  }

  fun accept(editor: Editor) {
    currentText?.let {
      WriteCommandAction.runWriteCommandAction(editor.project) {
        editor.document.insertString(currentOffset!!, it)
        editor.caretModel.moveToOffset(currentOffset!! + it.length)
      }
      currentText = null
      currentOffset = null
      currentInlays.forEach(Disposer::dispose)
      currentInlays = mutableListOf()
    }
  }

  fun dismiss() {
    currentText?.let {
      currentText = null
      currentOffset = null
      currentInlays.forEach(Disposer::dispose)
      currentInlays = mutableListOf()
    }
  }

  private fun addInlayLine(editor: Editor, offset: Int, line: String, index: Int) {
    val renderer = object : EditorCustomElementRenderer {
      override fun calcWidthInPixels(inlay: Inlay<*>): Int {
        // FIXME: Calc width?
        return 1
      }

      override fun paint(inlay: Inlay<*>, graphics: Graphics, targetRect: Rectangle, textAttributes: TextAttributes) {
        graphics.color = JBColor.GRAY
        graphics.drawString(line, targetRect.x, targetRect.y + inlay.editor.ascent)
      }
    }
    val inlay = if (index == 0) {
      editor.inlayModel.addInlineElement(offset, true, renderer)
    } else {
      editor.inlayModel.addBlockElement(offset, true, false, -index, renderer)
    }
    inlay?.let {
      currentInlays.add(it)
    }
  }
}