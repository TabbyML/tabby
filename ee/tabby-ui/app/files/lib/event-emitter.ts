import mitt from 'mitt'

type CodeBrowserQuickAction = 'explain' | 'generate_unittest' | 'generate_doc'
type LineMenuAction = 'copy_line' | 'copy_permalink'

type QuickActionEventPayload = {
  action: CodeBrowserQuickAction
  code: string
  language?: string
  path?: string
  lineFrom?: number
  lineTo?: number
}

type LineMenuActionEventPayload = {
  action: LineMenuAction
}

type CodeBrowserQuickActionEvents = {
  code_browser_quick_action: QuickActionEventPayload
  line_menu_action: LineMenuActionEventPayload
}

const emitter = mitt<CodeBrowserQuickActionEvents>()

export type {
  CodeBrowserQuickAction,
  QuickActionEventPayload,
  LineMenuAction,
  LineMenuActionEventPayload
}
export { emitter }
