import axios from "axios"

import React, { useRef, useEffect } from "react"
import Editor, { useMonaco } from "@monaco-editor/react"

let TabbyServerURL = "https://playground.app.tabbyml.com";

export default function MonacoEditor(props) {
  function beforeMount(monaco) {
    // Setup theming
    const LIGHT_BACKGROUND = getColor(
      "--ifm-background-color"
    );

    monaco.editor.defineTheme("tabby-light", {
      base: "vs",
      inherit: true,
      rules: [],
      colors: {
        "editor.background": LIGHT_BACKGROUND,
      }
    });

    monaco.languages.registerInlineCompletionsProvider(
      { pattern: "**" },
      new CompletionProvider(monaco)
    )

    monaco.editor.registerCommand(
      "acceptTabbyCompletion",
      (accessor, id, index) => {
        logAction(id, index, "select")
      }
    )
  }

  return (
    <Editor
      beforeMount={beforeMount}
      theme="tabby-light"
      defaultLanguage="python"
      {...props}
    />
  )
}

class CompletionProvider {
  constructor(monaco) {
    this.monaco = monaco
    this.latestTimestamp = 0
  }

  async provideInlineCompletions(document, position, context, token) {
    const segments = this.getSegments(document, position)
    const emptyResponse = Promise.resolve({ items: [] })

    if (this.isNil(segments.prefix)) {
      console.debug("Prefix is empty, skipping")
      return emptyResponse
    }

    const currentTimestamp = Date.now()
    this.latestTimestamp = currentTimestamp

    await this.sleep(500)
    if (currentTimestamp < this.latestTimestamp) {
      return emptyResponse
    }

    if (this.pendingRequest) {
      this.pendingRequest.cancelToken.cancel()
      this.pendingRequest = null
    }
    let response
    try {
      response = await this.requestCompletion(segments)
    } catch (err) {
      if (err.code !== "ERR_CANCELED") {
        console.error("error", err)
      }
      return emptyResponse
    }
    const hasSuffixParen = this.hasSuffixParen(document, position)
    const replaceRange = hasSuffixParen
      ? new this.monaco.Range(
          position.lineNumber,
          position.column,
          position.lineNumber,
          position.column + 1
        )
      : new this.monaco.Range(
          position.lineNumber,
          position.column,
          position.lineNumber,
          position.column
        )
    const items = this.toInlineCompletions(response.data, replaceRange)
    return Promise.resolve({ data: response.data, items })
  }

  handleItemDidShow(completions, item) {
    logAction(completions.data.id, item.choice.index, "view")
  }

  freeInlineCompletions() {}

  getSegments(document, position) {
    const firstLine = Math.max(position.lineNumber - 120, 1)
    const prefixRange = new this.monaco.Range(
      firstLine,
      1,
      position.lineNumber,
      position.column
    )
    const lastLine = Math.min(position.lineNumber + 120, document.getLineCount())
    const suffixRange = new this.monaco.Range(
      position.lineNumber,
      position.column,
      lastLine,
      document.getLineLength(lastLine) + 1
    )
    return {
      prefix: document.getValueInRange(prefixRange),
      suffix: document.getValueInRange(suffixRange),
    }
  }

  isNil(value) {
    return value === undefined || value === null || value.length === 0
  }

  sleep(milliseconds) {
    return new Promise((r) => setTimeout(r, milliseconds))
  }

  async requestCompletion(segments) {
    const cancelToken = axios.CancelToken.source()
    this.pendingRequest = {
      promise: axios.post(
        `${TabbyServerURL}/v1/completions`,
        {
          language: "python",
          segments,
        },
        {
          cancelToken: cancelToken.token,
        }
      ),
      cancelToken,
    }
    const response = await this.pendingRequest.promise
    this.pendingRequest = null
    return response
  }

  toInlineCompletions(value, range) {
    return (
      value.choices.map((choice) => ({
        range,
        insertText: choice.text,
        choice,
        command: {
          id: "acceptTabbyCompletion",
          arguments: [value.id, choice.index],
        },
      })) || []
    )
  }

  hasSuffixParen(document, position) {
    const suffix = document.getValueInRange(
      new this.monaco.Range(
        position.lineNumber,
        position.column,
        position.lineNumber,
        position.column + 1
      )
    )
    return ")]}".indexOf(suffix) > -1
  }
}

function logAction(completion_id, choice_index, type) {
  axios.post(`${TabbyServerURL}/v1/events`, {
    type,
    completion_id,
    choice_index,
  })
}

function getColor(property) {
  const styles = getComputedStyle(document.documentElement);
  // Weird chrome bug, returns " #ffffff " instead of "#ffffff", see: https://github.com/cloud-annotations/docusaurus-openapi/issues/144
  const color = styles.getPropertyValue(property).trim();
  if (color.length === 4) {
    // change hex short codes like "#fff" to "#ffffff"
    // to fix: https://github.com/cloud-annotations/docusaurus-openapi/issues/183
    let res = "#"; // prepend #
    for (const c of color.substring(1)) {
      res += c + c; // duplicate each char
    }
    return res;
  }
  return color;
}

