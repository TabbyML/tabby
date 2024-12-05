import { ReactNode, useContext, useEffect, useState } from 'react'
import { Element } from 'react-markdown/lib/ast-to-react'
import { SymbolInfo } from 'tabby-chat-panel/index'

import { cn } from '@/lib/utils'

import { CodeBlock } from '../ui/codeblock'
import { MessageMarkdownContext } from './markdown-context'

export interface CodeElementProps {
  node: Element
  inline?: boolean
  className?: string
  children: ReactNode & ReactNode[]
}

/**
 * Code element in Markdown AST.
 */
export function CodeElement({
  inline,
  className,
  children,
  ...props
}: CodeElementProps) {
  const {
    onLookupSymbol,
    canWrapLongLines,
    onApplyInEditor,
    onCopyContent,
    supportsOnApplyInEditorV2,
    activeSelection,
    onNavigateToContext
  } = useContext(MessageMarkdownContext)

  const [symbolLocation, setSymbolLocation] = useState<SymbolInfo | undefined>(
    undefined
  )

  const keyword = children[0]?.toString()

  useEffect(() => {
    const lookupSymbol = async () => {
      if (!inline || !onLookupSymbol || !keyword) return

      const symbolInfo = await onLookupSymbol(
        activeSelection?.filepath ? [activeSelection?.filepath] : [],
        keyword
      )
      setSymbolLocation(symbolInfo)
    }

    lookupSymbol()
  }, [inline, keyword, onLookupSymbol, activeSelection?.filepath])

  if (children.length) {
    if (children[0] === '▍') {
      return <span className="mt-1 animate-pulse cursor-default">▍</span>
    }
    children[0] = (children[0] as string).replace('`▍`', '▍')
  }

  if (inline) {
    const isClickable = Boolean(symbolLocation)

    const handleClick = () => {
      if (!isClickable || !symbolLocation || !onNavigateToContext) return

      onNavigateToContext(
        {
          filepath: symbolLocation.targetFile,
          range: {
            start: symbolLocation.targetLine,
            end: symbolLocation.targetLine
          },
          git_url: '',
          content: '',
          kind: 'file'
        },
        {
          openInEditor: true
        }
      )
    }

    return (
      <code
        className={cn(
          className,
          isClickable
            ? 'cursor-pointer transition-colors hover:bg-muted/50'
            : ''
        )}
        onClick={handleClick}
        {...props}
      >
        {children}
      </code>
    )
  }
  const match = /language-(\w+)/.exec(className || '')
  return (
    <CodeBlock
      key={Math.random()}
      language={(match && match[1]) || ''}
      value={String(children).replace(/\n$/, '')}
      onApplyInEditor={onApplyInEditor}
      onCopyContent={onCopyContent}
      canWrapLongLines={canWrapLongLines}
      supportsOnApplyInEditorV2={supportsOnApplyInEditorV2}
    />
  )
}
