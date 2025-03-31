import { ElementType, FC, memo } from 'react'
import ReactMarkdown, { Components, Options } from 'react-markdown'

import { MARKDOWN_CUSTOM_TAGS } from '@/lib/constants'

type ExtendedOptions = Omit<Options, 'components'> & {
  components: Components & {
    // for custom html tags rendering
    [Tag in (typeof MARKDOWN_CUSTOM_TAGS)[number]]?: ElementType
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
