import Link from 'next/link'

import { cn } from '@/lib/utils'
import { IconExternalLink } from '@/components/ui/icons'

export const RepositoryHeader = ({ className }: { className?: string }) => {
  return (
    <div className={cn('min-h-8 mb-4 flex items-center gap-4', className)}>
      <div className="flex-1 text-sm text-muted-foreground">
        Tabby supports connecting to Git repositories and uses these
        repositories as a context to enhance performance of large language model.
        <Link
          target="_blank"
          className="ml-2 inline-flex cursor-pointer flex-row items-center text-primary hover:underline"
          href="https://tabby.tabbyml.com/blog/2023/10/16/repository-context-for-code-completion"
        >
          Learn more
          <IconExternalLink />
        </Link>
      </div>
    </div>
  )
}
