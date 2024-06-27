package com.tabbyml.intellijtabby.completion

import com.intellij.openapi.application.invokeLater
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
import java.awt.Font
import java.awt.Graphics
import java.awt.Rectangle

class InlineCompletionRenderer {
  private val logger = Logger.getInstance(InlineCompletionRenderer::class.java)

  data class RenderingContext(
    val id: String,
    val editor: Editor,
    val offset: Int,
    val completionItem: InlineCompletionItem,
    val inlays: List<Inlay<*>>,
    val markups: List<RangeHighlighter>,
    val displayAt: Long,
  ) {
    fun calcElapsed(): Long {
      return System.currentTimeMillis() - displayAt
    }
  }

  var current: RenderingContext? = null
    private set

  fun show(
    editor: Editor, offset: Int, completion: InlineCompletionItem, callback: (context: RenderingContext) -> Unit = {}
  ) {
    invokeLater {
      current?.let {
        it.inlays.forEach(Disposer::dispose)
        it.markups.forEach { markup ->
          it.editor.markupModel.removeHighlighter(markup)
        }
        current = null
      }

      if (editor.caretModel.offset != offset) {
        return@invokeLater
      }

      logger.debug("Showing inline completion at $offset: $completion")

      val cmplId = completion.data?.eventId?.completionId?.replace("cmpl-", "") ?: "noCmplId"
      val displayAt = System.currentTimeMillis()
      val id = "view-${cmplId}-at-${displayAt}"

      val prefixReplaceLength = offset - completion.replaceRange.start
      val suffixReplaceLength = completion.replaceRange.end - offset
      val text = completion.insertText.substring(prefixReplaceLength)
      if (text.isEmpty()) {
        // Nothing to display
        return@invokeLater
      }
      val currentLineNumber = editor.document.getLineNumber(offset)
      val currentLineEndOffset = editor.document.getLineEndOffset(currentLineNumber)
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
      val context = RenderingContext(id, editor, offset, completion, inlays, markups, displayAt)
      current = context
      callback(context)
    }
  }

  fun hide() {
    current?.let {
      invokeLater {
        it.inlays.forEach(Disposer::dispose)
        it.markups.forEach { markup ->
          it.editor.markupModel.removeHighlighter(markup)
        }
      }
      current = null
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
        return editor.colorsScheme.getFont(EditorFontType.ITALIC).let {
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
