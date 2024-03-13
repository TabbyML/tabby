import React from 'react'
import { defaultHighlightStyle, syntaxHighlighting } from '@codemirror/language'
import {
  Annotation,
  EditorState,
  Extension,
  StateEffect
} from '@codemirror/state'
import { oneDarkHighlightStyle, oneDarkTheme } from '@codemirror/theme-one-dark'
import { EditorView } from '@codemirror/view'
import {
  loadLanguage,
  type LanguageName
} from '@uiw/codemirror-extensions-langs'
import { compact } from 'lodash-es'

import { basicSetup } from '@/components/codemirror/basic-setup'

import './style.css'

interface CodeMirrorEditorProps {
  value?: string
  language: LanguageName | string
  readonly?: boolean
  theme?: string
  height?: string
  width?: string
  extensions?: Extension[]
}

const External = Annotation.define<boolean>()

const CodeMirrorEditor: React.FC<CodeMirrorEditorProps> = ({
  value,
  theme,
  language,
  readonly = true,
  extensions: propsExtensions,
  height = null,
  width = null
}) => {
  const ref = React.useRef<HTMLDivElement>(null)
  const editor = React.useRef<EditorView | null>(null)

  const defaultThemeOption = EditorView.theme({
    '&': {
      height,
      width,
      outline: 'none !important',
      background: 'hsl(var(--background))'
    },
    '& .cm-scroller': {
      height: '100% !important'
    },
    '& .cm-gutters': {
      background: 'hsl(var(--background))'
    }
  })
  const extensions = [
    defaultThemeOption,
    basicSetup,
    EditorState.readOnly.of(readonly)
  ]

  const getExtensions = (): Extension[] => {
    let result = compact([...extensions, loadLanguage(getLanguage(language))])
    if (theme === 'dark') {
      result.push(oneDarkTheme)
      result.push(syntaxHighlighting(oneDarkHighlightStyle))
    } else {
      result.push(syntaxHighlighting(defaultHighlightStyle))
    }

    if (Array.isArray(propsExtensions)) {
      result = result.concat(propsExtensions)
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

  // refresh extension
  React.useEffect(() => {
    if (editor.current) {
      editor.current.dispatch({
        effects: StateEffect.reconfigure.of(getExtensions())
      })
    }
  }, [height, width, theme, language, propsExtensions])

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
          annotations: [External.of(true)]
        })
      }
    }

    resetValue()
  }, [value])

  return <div className="codemirror-editor h-full" ref={ref}></div>
}

function getLanguage(lang: LanguageName | string, ext?: string) {
  switch (lang) {
    case 'javascript-typescript':
      return 'tsx'
    case 'shellscript':
    case 'bash':
      return 'shell'
    default:
      return lang as LanguageName
  }
}

function SpaceDisplay({ spaceLength }: { spaceLength: number }) {
  const spaces = Array(spaceLength).fill('&nbsp;').join('')

  return <p dangerouslySetInnerHTML={{ __html: spaces }}></p>
}

export default CodeMirrorEditor
