import axios from "axios"
import { useRenderData } from "streamlit-component-lib-react-hooks"

import React, { useEffect } from "react"
import Monaco, { useMonaco } from "@monaco-editor/react"

const PythonParseJSON = `def parse_json_lines(filename: str) -> List[Any]:
    output = []
    with open(filename, "r", encoding="utf-8") as f:
`

export default function MonacoEditor() {
  const renderData = useRenderData()

  ;(window as any).tabbyServerURL = renderData.args.tabby_server_url

  const monaco = useMonaco()
  useEffect(() => {
    if (!monaco) return
    monaco.languages.registerInlineCompletionsProvider(
      { pattern: "**" },
      new CompletionProvider(monaco)
    )
  }, [monaco])

  return (
    <div style={{ height: 400 }}>
      <Monaco
        theme="vs-dark"
        defaultLanguage="python"
        defaultValue={PythonParseJSON}
      />
    </div>
  )
}

class CompletionProvider {
  private monaco: any
  private latestTimestamp: number
  private pendingRequest: any

  constructor(monaco: any) {
    this.monaco = monaco
    this.latestTimestamp = 0
  }

  async provideInlineCompletions(
    document: any,
    position: any,
    context: any,
    token: any
  ) {
    const prompt = this.getPrompt(document, position)
    const emptyResponse = Promise.resolve({ items: [] })

    if (this.isNil(prompt)) {
      console.debug("Prompt is empty, skipping")
      return emptyResponse
    }

    const currentTimestamp = Date.now()
    this.latestTimestamp = currentTimestamp

    await this.sleep(500)
    if (this.pendingRequest) await this.pendingRequest
    if (currentTimestamp < this.latestTimestamp) {
      return emptyResponse
    }

    const response = await this.callTabbyApi(currentTimestamp, prompt)
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
    return Promise.resolve({ items })
  }

  freeInlineCompletions() {}

  getPrompt(document: any, position: any): string {
    const firstLine = Math.max(position.lineNumber - 120, 0)

    const range = new this.monaco.Range(
      firstLine,
      0,
      position.lineNumber,
      position.column
    )
    return document.getValueInRange(range)
  }

  isNil(value: any) {
    return value === undefined || value === null || value.length === 0
  }

  sleep(milliseconds: number) {
    return new Promise((r) => setTimeout(r, milliseconds))
  }

  async callTabbyApi(timestamp: number, prompt: string) {
    const request = (this.pendingRequest = axios.post(
      `${(window as any).tabbyServerURL}/v1/completions`,
      {
        prompt,
      }
    ))
    const response = await request
    this.pendingRequest = null
    return response
  }

  toInlineCompletions(value: any, range: any) {
    return (
      value.choices
        ?.map((choice: any) => choice.text)
        .map((text: string) => ({
          range,
          text,
        })) || []
    )
  }

  hasSuffixParen(document: any, position: any): boolean {
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
