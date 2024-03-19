import React from 'react'
import rehypeRaw from 'rehype-raw'
import rehypeSanitize from 'rehype-sanitize'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'

import { MemoizedReactMarkdown } from '@/components/markdown'

interface MarkdownViewProps {
  value: string
}

const MarkdownView: React.FC<MarkdownViewProps> = ({ value }) => {
  return (
    <div className="px-10 py-4">
      <MemoizedReactMarkdown
        className="prose-full-width prose break-words dark:prose-invert prose-h1:border-b prose-h1:pb-2 prose-h2:border-b prose-h2:pb-2 prose-p:leading-relaxed prose-a:text-primary prose-img:inline prose-img:w-auto prose-img:max-w-full"
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeRaw, rehypeSanitize]}
      >
        {value}
      </MemoizedReactMarkdown>
    </div>
  )
}

export default MarkdownView
