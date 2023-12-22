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
import com.intellij.openapi.editor.markup.HighlighterLayer
import com.intellij.openapi.editor.markup.HighlighterTargetArea
import com.intellij.openapi.editor.markup.RangeHighlighter
import com.intellij.openapi.editor.markup.TextAttributes
import com.intellij.openapi.util.Disposer
import com.intellij.openapi.util.TextRange
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
    val inlays: List<Inlay<*>>,
    val markups: List<RangeHighlighter>,
  )

  var shownInlineCompletion: InlineCompletion? = null
    private set

  fun show(editor: Editor, offset: Int, completion: Agent.CompletionResponse) {
    dismiss()
    if (completion.choices.isEmpty()) {
      return
    }
    invokeLater {
      if (editor.caretModel.offset != offset) {
        return@invokeLater
      }

      // only support multiple choices for now
      val choice = completion.choices.first()
      logger.info("Showing inline completion at $offset: $choice")

      val prefixReplaceLength = offset - choice.replaceRange.start
      val suffixReplaceLength = choice.replaceRange.end - offset
      val text = choice.text.substring(prefixReplaceLength)
      if (text.isEmpty()) {
        return@invokeLater
      }
      val currentLineNumber = editor.document.getLineNumber(offset)
      val currentLineEndOffset = editor.document.getLineEndOffset(currentLineNumber)
      if (currentLineEndOffset - offset < suffixReplaceLength) {
        return@invokeLater
      }
      val currentLineSuffix = editor.document.getText(TextRange(offset, currentLineEndOffset))

      val textLines = text.lines().toMutableList()

      val inlays = mutableListOf<Inlay<*>>()
      val markups = mutableListOf<RangeHighlighter>()
      if (suffixReplaceLength == 0) {
        // No replace range to handle
        createInlayText(editor, textLines[0], offset, 0)?.let { inlays.add(it) }
        if (textLines.size > 1) {
          if (currentLineSuffix.isNotEmpty()) {
            markupReplaceText(editor, offset, currentLineEndOffset).let { markups.add(it) }
            textLines[textLines.lastIndex] += currentLineSuffix
          }
          textLines.forEachIndexed { index, line ->
            if (index > 0) {
              createInlayText(editor, line, offset, index)?.let { inlays.add(it) }
            }
          }
        }
      } else if (suffixReplaceLength == 1) {
        // Replace range contains one char
        val replaceChar = currentLineSuffix[0]
        // Insert part is substring of first line that before the char
        // Append part is substring of first line that after the char
        // If first line doesn't contain the char, insert part is full first line, append part is empty
        val insertPart = if (textLines[0].startsWith(replaceChar)) {
          ""
        } else {
          textLines[0].split(replaceChar).first()
        }
        val appendPart = if (insertPart.length < textLines[0].length) {
          textLines[0].substring(insertPart.length + 1)
        } else {
          ""
        }
        if (insertPart.isNotEmpty()) {
          createInlayText(editor, insertPart, offset, 0)?.let { inlays.add(it) }
        }
        if (appendPart.isNotEmpty()) {
          createInlayText(editor, appendPart, offset + 1, 0)?.let { inlays.add(it) }
        }
        if (textLines.size > 1) {
          if (currentLineSuffix.isNotEmpty()) {
            val startOffset = if (insertPart.length < textLines[0].length) {
              // First line contains the char
              offset + 1
            } else {
              // First line doesn't contain the char
              offset
            }
            markupReplaceText(editor, startOffset, currentLineEndOffset).let { markups.add(it) }
            textLines[textLines.lastIndex] += currentLineSuffix.substring(1)
          }
          textLines.forEachIndexed { index, line ->
            if (index > 0) {
              createInlayText(editor, line, offset, index)?.let { inlays.add(it) }
            }
          }
        }
      } else {
        // Replace range contains multiple chars
        // It's hard to match these chars in the insertion text, we just mark them up
        createInlayText(editor, textLines[0], offset, 0)?.let { inlays.add(it) }
        markupReplaceText(editor, offset, offset + suffixReplaceLength).let { markups.add(it) }
        if (textLines.size > 1) {
          if (currentLineSuffix.length > suffixReplaceLength) {
            markupReplaceText(editor, offset + suffixReplaceLength, currentLineEndOffset).let { markups.add(it) }
            textLines[textLines.lastIndex] += currentLineSuffix.substring(suffixReplaceLength)
          }
          textLines.forEachIndexed { index, line ->
            if (index > 0) {
              createInlayText(editor, line, offset, index)?.let { inlays.add(it) }
            }
          }
        }
      }

      shownInlineCompletion = InlineCompletion(editor, offset, completion, inlays, markups)
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

  enum class AcceptType {
    FULL_COMPLETION,
    NEXT_WORD,
    NEXT_LINE,
  }

  fun accept(type: AcceptType) {
    shownInlineCompletion?.let {
      val choice = it.completion.choices.first()
      logger.info("Accept inline completion at ${it.offset}: $type: $choice")

      val prefixReplaceLength = it.offset - choice.replaceRange.start
      val completionText = choice.text.substring(prefixReplaceLength)
      val text = when (type) {
        AcceptType.FULL_COMPLETION -> completionText
        AcceptType.NEXT_WORD -> {
          Regex("\\w+|\\W+").find(completionText)?.value ?: ""
        }

        AcceptType.NEXT_LINE -> {
          val lines = completionText.lines()
          if (lines.size <= 1) {
            completionText
          } else if (lines.first().isEmpty()) {
            lines.subList(0, 2).joinToString("\n")
          } else {
            lines.first()
          }
        }
      }
      WriteCommandAction.runWriteCommandAction(it.editor.project) {
        it.editor.document.deleteString(it.offset, choice.replaceRange.end)
        it.editor.document.insertString(it.offset, text)
        it.editor.caretModel.moveToOffset(it.offset + text.length)
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
            choiceIndex = choice.index,
            selectKind = when (type) {
              AcceptType.FULL_COMPLETION -> null
              AcceptType.NEXT_WORD, AcceptType.NEXT_LINE -> Agent.LogEventRequest.SelectKind.LINE
            }
          )
        )
      }
      shownInlineCompletion = null

      // Update inline completion after partial completion
      if (type == AcceptType.NEXT_LINE || type == AcceptType.NEXT_WORD) {
        invokeLater {
          val completionProvider = service<CompletionProvider>()
          completionProvider.provideCompletion(it.editor, it.offset + text.length, true)
        }
      }
    }
  }

  fun dismiss() {
    shownInlineCompletion?.let {
      invokeLater {
        it.inlays.forEach(Disposer::dispose)
        it.markups.forEach { markup ->
          it.editor.markupModel.removeHighlighter(markup)
        }
      }
      shownInlineCompletion = null
    }
  }

  private fun createInlayText(editor: Editor, text: String, offset: Int, lineOffset: Int): Inlay<*>? {
    val renderer = object : EditorCustomElementRenderer {
      override fun getContextMenuGroupId(inlay: Inlay<*>): String {
        return "Tabby.InlineCompletionContextMenu"
      }

      override fun calcWidthInPixels(inlay: Inlay<*>): Int {
        return maxOf(getWidth(inlay.editor, text), 1)
      }

      override fun paint(inlay: Inlay<*>, graphics: Graphics, targetRect: Rectangle, textAttributes: TextAttributes) {
        graphics.font = getFont(inlay.editor)
        graphics.color = JBColor.GRAY
        graphics.drawString(text, targetRect.x, targetRect.y + inlay.editor.ascent)
      }

      private fun getFont(editor: Editor): Font {
        return editor.colorsScheme.getFont(EditorFontType.PLAIN).let {
          UIUtil.getFontWithFallbackIfNeeded(it, text).deriveFont(editor.colorsScheme.editorFontSize)
        }
      }

      private fun getWidth(editor: Editor, line: String): Int {
        val font = getFont(editor)
        val metrics = FontInfo.getFontMetrics(font, FontInfo.getFontRenderContext(editor.contentComponent))
        return metrics.stringWidth(line)
      }
    }
    return if (lineOffset == 0) {
      editor.inlayModel.addInlineElement(offset, true, renderer)
    } else {
      editor.inlayModel.addBlockElement(offset, true, false, -lineOffset, renderer)
    }
  }

  private fun markupReplaceText(editor: Editor, startOffset: Int, endOffset: Int): RangeHighlighter {
    val textAttributes = TextAttributes().apply {
      foregroundColor = JBColor.background()
      backgroundColor = JBColor.background()
    }
    return editor.markupModel.addRangeHighlighter(
      startOffset, endOffset, HighlighterLayer.LAST + 1000, textAttributes, HighlighterTargetArea.EXACT_RANGE
    )
  }
}
