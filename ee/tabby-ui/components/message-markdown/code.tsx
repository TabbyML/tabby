import { ReactNode, useContext, useEffect } from 'react'
import { Element } from 'react-markdown/lib/ast-to-react'

import { cn } from '@/lib/utils'

import { CodeBlock } from '../ui/codeblock'
import { IconSquareChevronRight } from '../ui/icons'
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
    lookupSymbol,
    openInEditor,
    isStreaming,
    onApplyInEditor,
    onCopyContent,
    supportsOnApplyInEditorV2,
    symbolPositionMap,
    runShell
  } = useContext(MessageMarkdownContext)

  const keyword = children[0]?.toString()
  const symbolInfo = keyword ? symbolPositionMap.get(keyword) : undefined

  useEffect(() => {
    if (!inline || !lookupSymbol || !keyword) return
    lookupSymbol(keyword)
  }, [inline, keyword, lookupSymbol])

  if (children.length) {
    if (children[0] === '▍') {
      return <span className="mt-1 animate-pulse cursor-default">▍</span>
    }
    children[0] = (children[0] as string).replace('`▍`', '▍')
  }

  if (inline) {
    const isSymbolNavigable = Boolean(symbolInfo?.target)

    const handleClick = () => {
      if (isSymbolNavigable && openInEditor && symbolInfo?.target) {
        openInEditor(symbolInfo.target)
      }
    }

    return (
      <code
        className={cn('group/symbol', className, {
          symbol: !!lookupSymbol,
          'bg-muted leading-5 py-0.5': !!lookupSymbol && !isSymbolNavigable,
          'space-x-1 cursor-pointer hover:bg-muted/50 border whitespace-nowrap align-middle py-0.5':
            isSymbolNavigable
        })}
        onClick={handleClick}
        {...props}
      >
        {isSymbolNavigable && (
          <IconSquareChevronRight className="relative -top-px inline-block h-3.5 w-3.5 text-primary" />
        )}
        <span
          className={cn('whitespace-normal', {
            'group-hover/symbol:text-primary': isSymbolNavigable
          })}
        >
          {children}
        </span>
      </code>
    )
  }

  const match = /language-(\w+)/.exec(className || '')
  return (
    <CodeBlock
      language={(match && match[1]) || ''}
      value={String(children).replace(/\n$/, '')}
      onApplyInEditor={onApplyInEditor}
      onCopyContent={onCopyContent}
      isStreaming={isStreaming}
      supportsOnApplyInEditorV2={supportsOnApplyInEditorV2}
      runShell={runShell}
    />
  )
}
