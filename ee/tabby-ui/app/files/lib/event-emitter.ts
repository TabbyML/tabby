import mitt from 'mitt'
import { ChatCommand, EditorFileContext } from 'tabby-chat-panel/index'

type LineMenuAction = 'copy-line' | 'copy-permalink'

type LineMenuActionEventPayload = {
  action: LineMenuAction
}

type SelectionChangeEventPayload = EditorFileContext | null

type CodeBrowserEvents = {
  quick_action_command: ChatCommand
  line_menu_action: LineMenuActionEventPayload
  selection_change: SelectionChangeEventPayload
}

export const emitter = mitt<CodeBrowserEvents>()

export type { LineMenuAction, LineMenuActionEventPayload }
