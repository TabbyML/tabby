import mitt from 'mitt'

type CodeBrowserQuickAction = 'explain' | 'generate_unittest' | 'generate_doc'

type QuickActionEventPayload = {
  action: CodeBrowserQuickAction
  code: string
  language?: string
}

type CodeBrowserQuickActionEvents = {
  code_browser_quick_action: QuickActionEventPayload
}

const emitter = mitt<CodeBrowserQuickActionEvents>()

export type { CodeBrowserQuickAction, QuickActionEventPayload }
export { emitter }
