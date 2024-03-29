import mitt from 'mitt'

type Events = {
  code_browser_quick_action: string
}

export const emitter = mitt<Events>()
