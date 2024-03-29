import mitt from 'mitt'

type CodeBrowserQuickAction = 'explain' | 'generate_unittest' | 'generate_doc'

type CodeBrowserQuickActionEvents = {
  code_browser_quick_action: { action: CodeBrowserQuickAction, payload: string }
}

const emitter = mitt<CodeBrowserQuickActionEvents>()

export type { CodeBrowserQuickAction }
export { emitter }
