import { ElementType, FC, memo } from 'react'
import ReactMarkdown, { Components, Options } from 'react-markdown'

import {
  CUSTOM_HTML_BLOCK_TAGS,
  CUSTOM_HTML_INLINE_TAGS
} from '@/lib/constants'

type CustomTag =
  | (typeof CUSTOM_HTML_BLOCK_TAGS)[number]
  | (typeof CUSTOM_HTML_INLINE_TAGS)[number]

type ExtendedOptions = Omit<Options, 'components'> & {
  components?: Components & {
    // for custom html tags rendering
    [Tag in CustomTag]?: ElementType
  }
}

const Markdown = ({ className, ...props }: ExtendedOptions) => (
  <div className={className}>
    <ReactMarkdown {...props} />
  </div>
)

export const MemoizedReactMarkdown: FC<ExtendedOptions> = memo(
  Markdown,
  (prevProps, nextProps) =>
    prevProps.children === nextProps.children &&
    prevProps.className === nextProps.className
)
