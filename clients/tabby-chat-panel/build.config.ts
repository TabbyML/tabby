import { defineBuildConfig } from 'unbuild'

export default defineBuildConfig({
  entries: [
    'src/index',
    'src/react',
    'src/createThread',
    'src/createThreadInsideIframe',
  ],
  declaration: true,
  clean: true,
  rollup: {
    emitCJS: true,
  },
})
