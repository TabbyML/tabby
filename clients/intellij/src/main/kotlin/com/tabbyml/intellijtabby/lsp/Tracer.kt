package com.tabbyml.intellijtabby.lsp

import com.intellij.openapi.diagnostic.Logger
import java.io.Writer

class Tracer : Writer() {
  private val logger = Logger.getInstance(Tracer::class.java)

  override fun write(cbuf: CharArray, off: Int, len: Int) {
    logger.trace(String(cbuf.sliceArray(IntRange(off, off + len - 1))))
  }

  override fun flush() {
    // nothing
  }

  override fun close() {
    // nothing
  }
}