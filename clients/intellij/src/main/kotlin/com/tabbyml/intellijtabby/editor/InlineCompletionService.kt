package com.tabbyml.intellijtabby.editor

import com.intellij.openapi.application.invokeLater
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

  data class InlineCompletion(val editor: Editor, val text: String, val offset: Int, val inlays: List<Inlay<*>>)

  var shownInlineCompletion: InlineCompletion? = null
    private set

  fun show(editor: Editor, offset: Int, completion: Agent.CompletionResponse) {
    dismiss()
    if (completion.choices.isEmpty()) {
      return
    }
    invokeLater {
      val text = completion.choices.first().text
      logger.info("Showing inline completion at $offset: $text")
      val lines = text.split("\n")
      val inlays = lines
        .mapIndexed { index, line -> createInlayLine(editor, offset, line, index) }
        .filterNotNull()
      shownInlineCompletion = InlineCompletion(editor, text, offset, inlays)
    }
  }

  fun accept() {
    shownInlineCompletion?.let {
      logger.info("Accept inline completion at ${it.offset}: ${it.text}")
      WriteCommandAction.runWriteCommandAction(it.editor.project) {
        it.editor.document.insertString(it.offset, it.text)
        it.editor.caretModel.moveToOffset(it.offset + it.text.length)
      }
      invokeLater {
        it.inlays.forEach(Disposer::dispose)
      }
      shownInlineCompletion = null
    }
  }

  fun dismiss() {
    shownInlineCompletion?.let {
      invokeLater {
        it.inlays.forEach(Disposer::dispose)
      }
      shownInlineCompletion = null
    }
  }

  private fun createInlayLine(editor: Editor, offset: Int, line: String, index: Int): Inlay<*>? {
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
    return if (index == 0) {
      editor.inlayModel.addInlineElement(offset, true, renderer)
    } else {
      editor.inlayModel.addBlockElement(offset, true, false, -index, renderer)
    }
  }
}