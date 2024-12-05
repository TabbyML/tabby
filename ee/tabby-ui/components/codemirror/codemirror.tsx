import React from 'react'
import {
  defaultHighlightStyle,
  LanguageSupport,
  syntaxHighlighting
} from '@codemirror/language'
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
import { graphqlLanguage } from 'cm6-graphql'
import { compact } from 'lodash-es'

import { basicSetup } from '@/components/codemirror/basic-setup'

import './style.css'

import { cn } from '@/lib/utils'

interface CodeMirrorEditorProps {
  value?: string
  language: LanguageName | string
  readonly?: boolean
  theme?: string
  height?: string
  width?: string
  extensions?: Extension[]
  viewDidUpdate?: (view: EditorView | null) => void
  className?: string
}

export interface CodeMirrorEditorRef {
  getView: () => EditorView | null
}

const External = Annotation.define<boolean>()

const CodeMirrorEditor = React.forwardRef<
  CodeMirrorEditorRef,
  CodeMirrorEditorProps
>((props, ref) => {
  const {
    value,
    theme,
    language,
    readonly = true,
    extensions: propsExtensions,
    height = null,
    width = null,
    viewDidUpdate,
    className
  } = props

  const initialized = React.useRef(false)
  const containerRef = React.useRef<HTMLDivElement>(null)
  const [editorView, setEditorView] = React.useState<EditorView | null>(null)

  const defaultThemeOption = EditorView.theme({
    '&': {
      height,
      width,
      outline: 'none !important',
      background: 'hsl(var(--background))'
    },
    '&.cm-focused': {
      outline: 'none !important'
    },
    '& .cm-scroller': {
      height: '100% !important',
      outline: 'none'
    },
    '& .cm-gutters': {
      background: 'hsl(var(--background))'
    },
    '&.cm-focused .cm-selectionLayer .cm-selectionBackground': {
      backgroundColor: 'hsl(var(--cm-selection-bg)) !important'
    },
    '.cm-selectionLayer .cm-selectionBackground': {
      backgroundColor: 'hsl(var(--cm-selection-bg)) !important'
    }
  })

  const extensions = [
    defaultThemeOption,
    basicSetup,
    EditorView.baseTheme({
      '.cm-line': {
        lineHeight: '20px'
      },
      '.cm-scroller': {
        fontSize: '14px'
      },
      '.cm-gutters': {
        backgroundColor: 'transparent',
        borderRight: 'none'
      }
    }),
    EditorState.readOnly.of(readonly)
  ]

  const languageHandler = (language: string) => {
    if (language === 'graphql') {
      return new LanguageSupport(graphqlLanguage)
    }
    return loadLanguage(getLanguage(language))
  }

  const getExtensions = (): Extension[] => {
    let result = compact([...extensions, languageHandler(language)])
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
      if (initialized.current) return

      if (containerRef.current) {
        initialized.current = true
        let startState = EditorState.create({
          doc: value,
          extensions: getExtensions()
        })

        const _view = new EditorView({
          state: startState,
          parent: containerRef.current
        })
        setEditorView(_view)
      }
    }

    initEditor()
  }, [])

  // refresh extension
  React.useEffect(() => {
    if (editorView) {
      editorView.dispatch({
        effects: StateEffect.reconfigure.of(getExtensions())
      })
    }
  }, [height, width, theme, language, propsExtensions])

  React.useEffect(() => {
    const resetValue = () => {
      if (value === undefined || !editorView) return

      const currentValue = editorView ? editorView.state.doc.toString() : ''
      if (editorView && value !== currentValue) {
        editorView.dispatch({
          changes: { from: 0, to: currentValue.length, insert: value || '' },
          annotations: [External.of(true)]
        })
      }
    }

    resetValue()
  }, [value])

  React.useEffect(
    () => () => {
      if (editorView) {
        editorView.destroy()
        setEditorView(null)
      }
    },
    []
  )

  React.useEffect(() => {
    viewDidUpdate?.(editorView)
  }, [editorView])

  React.useImperativeHandle(
    ref,
    () => {
      return {
        getView: () => editorView
      }
    },
    [editorView]
  )

  return (
    <div
      className={cn('codemirror-editor h-full', className)}
      ref={containerRef}
    ></div>
  )
})

CodeMirrorEditor.displayName = 'CodeMirrorEditor'

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

export default CodeMirrorEditor
