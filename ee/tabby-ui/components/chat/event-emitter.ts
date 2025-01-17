import mitt from 'mitt'

type Events = {
  file_mention_update: void
}

export const emitter = mitt<Events>()
