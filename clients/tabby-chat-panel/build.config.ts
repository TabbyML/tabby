import { defineBuildConfig } from 'unbuild'

export default defineBuildConfig({
  entries: [
    'src/index',
    'src/react',
  ],
  declaration: true,
  clean: true,
  rollup: {
    emitCJS: true,
  },
})
