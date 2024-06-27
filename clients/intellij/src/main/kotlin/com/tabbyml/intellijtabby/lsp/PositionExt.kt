package com.tabbyml.intellijtabby.lsp

import com.intellij.openapi.editor.Document
import org.eclipse.lsp4j.Position

fun positionInDocument(document: Document, offset: Int): Position {
  val line = document.getLineNumber(offset)
  val character = offset - document.getLineStartOffset(line)
  return Position(line, character)
}

fun offsetInDocument(document: Document, position: Position): Int {
  return document.getLineStartOffset(position.line) + position.character
}
