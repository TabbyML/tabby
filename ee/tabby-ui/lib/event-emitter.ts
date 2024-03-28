import mitt from 'mitt'

type CodeBrowserQuickAction =
  | 'explain_detail'
  | 'generate_unit-test'
  | 'generate_doc'

type Events = {
  code_browser_quick_action: CodeBrowserQuickAction
}

const emitter = mitt<Events>()

export type { CodeBrowserQuickAction }
export { emitter }
