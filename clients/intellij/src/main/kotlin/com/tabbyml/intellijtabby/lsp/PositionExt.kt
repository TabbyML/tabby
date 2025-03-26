package com.tabbyml.intellijtabby.lsp

import com.intellij.openapi.editor.Document
import org.eclipse.lsp4j.Position

fun positionInDocument(document: Document, offset: Int): Position {
  val line = document.getLineNumber(offset)
  val character = offset - document.getLineStartOffset(line)
  return Position(line, character)
}

fun offsetInDocument(document: Document, position: Position): Int {
  if (position.line < 0) {
    return position.character
  }
  if (position.line >= document.lineCount) {
    return document.getLineEndOffset(document.lineCount - 1)
  }
  return document.getLineStartOffset(position.line) + position.character
}
