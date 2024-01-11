import React from 'react'
import { defaultHighlightStyle, syntaxHighlighting } from '@codemirror/language'
import { EditorState, Extension, StateEffect } from '@codemirror/state'
import { oneDarkHighlightStyle, oneDarkTheme } from '@codemirror/theme-one-dark'
import { EditorView } from '@codemirror/view'
import { PopoverTrigger } from '@radix-ui/react-popover'
import {
  loadLanguage,
  type LanguageName
} from '@uiw/codemirror-extensions-langs'
import { compact } from 'lodash-es'
import { createRoot } from 'react-dom/client'

import { basicSetup } from '@/components/codemirror/basic-setup'
import { TCodeTag } from '@/app/browser/components/source-code-browser'

import { IconSymbolFunction } from '../ui/icons'
import { Popover, PopoverContent, PopoverPortal } from '../ui/popover'
import { displayCodeTagWidgets, setCodeTagData } from './code-tag-extension'

interface CodeMirrorEditorProps {
  value?: string
  language: LanguageName | string
  readonly?: boolean
  theme?: string
  tags?: TCodeTag[]
}

type GetExtensionsOptions = {
  excludeWidget?: boolean
}

export const CodeMirrorEditor: React.FC<CodeMirrorEditorProps> = ({
  value,
  theme,
  language,
  readonly = true,
  tags
}) => {
  const ref = React.useRef<HTMLDivElement>(null)
  const editor = React.useRef<EditorView | null>(null)
  const extensions = [basicSetup, EditorState.readOnly.of(readonly)]

  const getExtensions = (options?: GetExtensionsOptions): Extension[] => {
    let result = compact([...extensions, loadLanguage(getLanguage(language))])
    if (theme === 'dark') {
      result.push(oneDarkTheme)
      result.push(syntaxHighlighting(oneDarkHighlightStyle))
    } else {
      result.push(syntaxHighlighting(defaultHighlightStyle))
    }

    if (value && !options?.excludeWidget) {
      result.push(
        setCodeTagData(
          tags?.map(tag => {
            return {
              item: tag.syntax_type_name,
              range: { start: tag.name_range.start, end: tag.name_range.start }
            }
          }) || []
        ),
        displayCodeTagWidgets({
          createDecoration(container, { items, indent }) {
            const div = document.createElement('div')
            const root = createRoot(div)
            root.render(
              <div className="flex flex-nowrap items-center gap-1">
                {indent ? <SpaceDisplay spaceLength={indent.length} /> : null}
                {items.map((item, index) => {
                  return (
                    <Popover key={index}>
                      <PopoverTrigger>
                        <div className="flex items-center gap-1 rounded-sm border bg-secondary px-1">
                          <IconSymbolFunction />
                          <span>{item}</span>
                        </div>
                      </PopoverTrigger>
                      <PopoverPortal>
                        <PopoverContent side="bottom" align="start">
                          {item}
                        </PopoverContent>
                      </PopoverPortal>
                    </Popover>
                  )
                })}
              </div>
            )

            container.append(div)

            return {
              destroy() {
                div.remove()
              }
            }
          }
        })
      )
    }

    return result
  }

  React.useEffect(() => {
    const initEditor = () => {
      if (ref.current) {
        let startState = EditorState.create({
          doc: value,
          extensions
        })

        editor.current = new EditorView({
          state: startState,
          parent: ref.current
        })
      }
    }

    initEditor()

    return () => {
      if (editor.current) {
        editor.current.destroy()
      }
    }
  }, [])

  // refresh theme and language
  React.useEffect(() => {
    if (editor.current) {
      editor.current.dispatch({
        effects: StateEffect.reconfigure.of(getExtensions())
      })
    }
  }, [theme, language])

  React.useEffect(() => {
    const resetValue = () => {
      if (value === undefined || !editor.current) {
        return
      }
      const currentValue = editor.current
        ? editor.current.state.doc.toString()
        : ''
      if (editor.current && value !== currentValue) {
        editor.current.dispatch({
          changes: { from: 0, to: currentValue.length, insert: value || '' },
          effects: StateEffect.reconfigure.of(
            getExtensions({ excludeWidget: false })
          )
        })
      }
    }

    resetValue()
  }, [value])

  React.useEffect(() => {
    const dispatchTagWidget = () => {
      if (!value) return

      setTimeout(() => {
        editor.current!.dispatch({
          effects: StateEffect.reconfigure.of(getExtensions())
        })
      }, 200)
    }

    dispatchTagWidget()
  }, [tags])

  return <div ref={ref}></div>
}

function getLanguage(lang: LanguageName | string, ext?: string) {
  switch (lang) {
    case 'javascript-typescript':
      return 'tsx'
    case 'shellscript':
      return 'shell'
    default:
      return lang as LanguageName
  }
}

function SpaceDisplay({ spaceLength }: { spaceLength: number }) {
  const spaces = Array(spaceLength).fill('&nbsp;').join('')

  return <p dangerouslySetInnerHTML={{ __html: spaces }}></p>
}
