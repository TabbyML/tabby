import path from 'path'
import { has } from 'lodash-es'
import {isNil} from 'lodash-es'

import { Language as ProgrammingLanguage } from '@/lib/gql/generates/graphql'

import languages from './languages'
import languageColors from './language-colors.json'

const languageColorMapping: Record<string, string> = Object
  .entries(languageColors)
  .reduce((acc, cur) => {
    const [lan, color] = cur
    return { ...acc, [lan.toLocaleLowerCase()]: color }
  }, {})

// Fork from
// https://github.com/TomerAberbach/filename2prism/
export const filename2prism: (filename: string) => Array<string> = filename => {
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

  const result: string[] = []
  return result
    .concat(
      filenames[path.basename(filename)],
      extnames[path.extname(filename).substring(1)]
    )
    .filter(Boolean)
}

export const getLanguageDisplayName = (
  lan?: string,
  defaultLan?: string
): string => {
  const returnDefault = () => !isNil(defaultLan) ? defaultLan : 'Other'
  if (!lan) return returnDefault()

  const indexInSupportedLanguages = Object
    .values(ProgrammingLanguage)
    .map(lan => lan.toLocaleLowerCase())
    .indexOf(lan)
  if (indexInSupportedLanguages === -1) return returnDefault()

  const displayName = Object.values(ProgrammingLanguage)[indexInSupportedLanguages]
  const mapping: Record<string, string> = {
    csharp: 'C#',
    cpp: 'C++',
    javascript: 'JavaScript',
    typescript: 'TypeScript'
  }
  return mapping[displayName.toLocaleLowerCase()] || displayName
}

export const getLanguageColor = (lan: string): string | undefined => {
  return languageColorMapping[lan.toLowerCase()]
}
