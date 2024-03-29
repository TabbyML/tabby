import mitt from 'mitt'

type CodeBrowserQuickAction =
  | 'explain_detail'
  | 'generate_unittest'
  | 'generate_doc'

type CodeBrowserQuickActionEvents = {
  code_browser_quick_action: CodeBrowserQuickAction
}

const emitter = mitt<CodeBrowserQuickActionEvents>()

export type { CodeBrowserQuickAction }
export { emitter }
