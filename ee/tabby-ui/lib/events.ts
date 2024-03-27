import mitt from 'mitt'

type Events = {
  code_browser_action_prompt: string
}

export const emitter = mitt<Events>()
