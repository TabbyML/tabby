import { FC, memo } from 'react'
import ReactMarkdown, { Options } from 'react-markdown'

const Markdown = (props: Options) => (
  <ReactMarkdown linkTarget="_blank" {...props} />
)

export const MemoizedReactMarkdown: FC<Options> = memo(
  Markdown,
  (prevProps, nextProps) =>
    prevProps.children === nextProps.children &&
    prevProps.className === nextProps.className
)
