import { defineBuildConfig } from 'unbuild'

// FIXME(wwayne): add build:vscode & dev:vscode
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
