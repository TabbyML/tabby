package com.tabbyml.intellijtabby.editor

import com.intellij.openapi.application.invokeLater
import com.intellij.openapi.command.WriteCommandAction
import com.intellij.openapi.components.Service
import com.intellij.openapi.components.service
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.EditorCustomElementRenderer
import com.intellij.openapi.editor.Inlay
import com.intellij.openapi.editor.colors.EditorFontType
import com.intellij.openapi.editor.impl.FontInfo
import com.intellij.openapi.editor.markup.TextAttributes
import com.intellij.openapi.util.Disposer
import com.intellij.ui.JBColor
import com.intellij.util.ui.UIUtil
import com.tabbyml.intellijtabby.agent.Agent
import com.tabbyml.intellijtabby.agent.AgentService
import kotlinx.coroutines.launch
import java.awt.Font
import java.awt.Graphics
import java.awt.Rectangle


@Service
class InlineCompletionService {
  private val logger = Logger.getInstance(InlineCompletionService::class.java)

  data class InlineCompletion(
    val editor: Editor,
    val offset: Int,
    val completion: Agent.CompletionResponse,
    val text: String,
    val inlays: List<Inlay<*>>,
  )

  var shownInlineCompletion: InlineCompletion? = null
    private set

  fun show(editor: Editor, offset: Int, completion: Agent.CompletionResponse) {
    dismiss()
    if (completion.choices.isEmpty()) {
      return
    }
    invokeLater {
      // FIXME: support multiple choices
      val text = completion.choices.first().text
      logger.info("Showing inline completion at $offset: $text")
      val lines = text.split("\n")
      val inlays = lines
        .mapIndexed { index, line -> createInlayLine(editor, offset, line, index) }
        .filterNotNull()
      shownInlineCompletion = InlineCompletion(editor, offset, completion, text, inlays)
    }
    val agentService = service<AgentService>()
    agentService.scope.launch {
      agentService.postEvent(
        Agent.LogEventRequest(
          type = Agent.LogEventRequest.EventType.VIEW,
          completionId = completion.id,
          choiceIndex = completion.choices.first().index
        )
      )
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
      val agentService = service<AgentService>()
      agentService.scope.launch {
        agentService.postEvent(
          Agent.LogEventRequest(
            type = Agent.LogEventRequest.EventType.SELECT,
            completionId = it.completion.id,
            choiceIndex = it.completion.choices.first().index
          )
        )
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
        return maxOf(getWidth(inlay.editor, line), 1)
      }

      override fun paint(inlay: Inlay<*>, graphics: Graphics, targetRect: Rectangle, textAttributes: TextAttributes) {
        graphics.font = getFont(inlay.editor)
        graphics.color = JBColor.GRAY
        graphics.drawString(line, targetRect.x, targetRect.y + inlay.editor.ascent)
      }

      private fun getFont(editor: Editor): Font {
        return editor.colorsScheme.getFont(EditorFontType.PLAIN).let {
          UIUtil.getFontWithFallbackIfNeeded(it, line).deriveFont(editor.colorsScheme.editorFontSize)
        }
      }

      private fun getWidth(editor: Editor, line: String): Int {
        val font = getFont(editor)
        val metrics = FontInfo.getFontMetrics(font, FontInfo.getFontRenderContext(editor.contentComponent))
        return metrics.stringWidth(line)
      }
    }
    return if (index == 0) {
      editor.inlayModel.addInlineElement(offset, true, renderer)
    } else {
      editor.inlayModel.addBlockElement(offset, true, false, -index, renderer)
    }
  }
}