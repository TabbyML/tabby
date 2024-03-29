import Image from 'next/image'
import tabbyLogo from '@/assets/tabby.png'

import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { IconChevronUpDown } from '@/components/ui/icons'

import { CodeBrowserQuickAction, emitter } from '../../lib/event-emitter'

interface ActionBarWidgetProps extends React.HTMLAttributes<HTMLDivElement> {}

export const ActionBarWidget: React.FC<ActionBarWidgetProps> = ({
  className,
  ...props
}) => {
  const handleAction = (action: CodeBrowserQuickAction) => {
    emitter.emit('code_browser_quick_action', action)
  }

  return (
    <div
      className={cn(
        'mt-2 flex items-center gap-2 rounded-md border bg-background px-2 py-1',
        className
      )}
      {...props}
    >
      <Image src={tabbyLogo} width={32} alt="logo" />
      <Button
        size="sm"
        variant="outline"
        onClick={e => handleAction('explain')}
      >
        Explain
      </Button>
      <DropdownMenu modal={false}>
        <DropdownMenuTrigger asChild>
          <Button size="sm" variant="outline">
            Generate
            <IconChevronUpDown className="ml-1" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start">
          <DropdownMenuItem
            className="cursor-pointer"
            onSelect={() => handleAction('generate_unittest')}
          >
            Unit Test
          </DropdownMenuItem>
          <DropdownMenuItem
            className="cursor-pointer"
            onSelect={() => handleAction('generate_doc')}
          >
            Documentation
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}
