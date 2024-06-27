package com.tabbyml.intellijtabby.lsp

import com.intellij.lang.Language
import com.intellij.psi.PsiFile
import io.ktor.util.*


private val languageIdMap = mapOf(
  "ObjectiveC" to "objective-c",
  "ObjectiveC++" to "objective-cpp",
)
private val filetypeMap = mapOf(
  "py" to "python",
  "js" to "javascript",
  "cjs" to "javascript",
  "mjs" to "javascript",
  "jsx" to "javascriptreact",
  "ts" to "typescript",
  "tsx" to "typescriptreact",
  "kt" to "kotlin",
  "md" to "markdown",
  "cc" to "cpp",
  "cs" to "csharp",
  "m" to "objective-c",
  "mm" to "objective-cpp",
  "sh" to "shellscript",
  "zsh" to "shellscript",
  "bash" to "shellscript",
  "txt" to "plaintext",
)

// Language id: https://code.visualstudio.com/docs/languages/identifiers
fun PsiFile.getLanguageId(): String {
  return if (language != Language.ANY &&
    language.id.isNotBlank() &&
    language.id.toLowerCasePreservingASCIIRules() !in arrayOf("txt", "text", "textmate")
  ) {
    languageIdMap[language.id] ?: language.id
      .toLowerCasePreservingASCIIRules()
      .replace("#", "sharp")
      .replace("++", "pp")
      .replace(" ", "")
  } else {
    val ext = this.fileType.defaultExtension.ifBlank {
      this.virtualFile.name.substringAfterLast(".")
    }
    if (ext.isNotBlank()) {
      filetypeMap[ext] ?: ext.toLowerCasePreservingASCIIRules()
    } else {
      "plaintext"
    }
  }
}

