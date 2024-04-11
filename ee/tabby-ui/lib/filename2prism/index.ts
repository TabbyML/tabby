// Fork from
// https://github.com/TomerAberbach/filename2prism/

import path from 'path'
import { has } from 'lodash-es'

import languages from './languages'

const filenames: Record<string, string[]> = {}
const extnames: Record<string, string[]> = {}

for (const [alias, associations] of Object.entries(languages)) {
  for (const filename of associations.filenames) {
    if (!has(filenames, filename)) {
      filenames[filename] = []
    }

    filenames[filename].push(alias)
  }

  for (const extname of associations.extnames) {
    if (!has(extnames, extname)) {
      extnames[extname] = []
    }

    extnames[extname].push(alias)
  }
}

const filename2prism: (filename: string) => Array<string> = filename => {
  const result: string[] = []
  return result
    .concat(
      filenames[path.basename(filename)],
      extnames[path.extname(filename).substring(1)]
    )
    .filter(Boolean)
}

export default filename2prism
