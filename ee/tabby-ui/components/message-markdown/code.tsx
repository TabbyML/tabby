import { ReactNode, useContext } from 'react'
import { Element } from 'react-markdown/lib/ast-to-react'

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
    onNavigateSymbol,
    canWrapLongLines,
    onApplyInEditor,
    onCopyContent,
    supportsOnApplyInEditorV2,
    activeSelection
  } = useContext(MessageMarkdownContext)

  if (children.length) {
    if (children[0] === '▍') {
      return <span className="mt-1 animate-pulse cursor-default">▍</span>
    }
    children[0] = (children[0] as string).replace('`▍`', '▍')
  }

  const match = /language-(\w+)/.exec(className || '')

  if (inline) {
    if (!onNavigateSymbol) {
      return (
        <code className={className} {...props}>
          {children}
        </code>
      )
    }

    const keyword = children[0]?.toString()
    if (!keyword) {
      return (
        <code className={className} {...props}>
          {children}
        </code>
      )
    }

    const isClickable = Boolean(canWrapLongLines)

    const handleClick = () => {
      if (!isClickable) return
      if (onNavigateSymbol) {
        onNavigateSymbol(
          activeSelection?.filepath ? [activeSelection?.filepath] : [],
          keyword
        )
      }
    }

    return (
      <code
        className={cn(
          className,
          isClickable
            ? 'hover:bg-muted/50 cursor-pointer transition-colors'
            : ''
        )}
        onClick={handleClick}
        {...props}
      >
        {children}
      </code>
    )
  }

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
