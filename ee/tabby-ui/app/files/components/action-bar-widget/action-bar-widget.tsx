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

interface CompletionWidgetProps extends React.HTMLAttributes<HTMLDivElement> {}

export const ActionBarWidget: React.FC<CompletionWidgetProps> = ({
  className,
  ...props
}) => {
  return (
    <div
      className={cn(
        'flex items-center gap-2 rounded-md border bg-background px-2 py-1',
        className
      )}
      {...props}
    >
      <Image src={tabbyLogo} width={32} alt="logo" />
      <DropdownMenu modal={false}>
        <DropdownMenuTrigger asChild>
          <Button size="sm" variant="outline">
            Explain
            <IconChevronUpDown className="ml-1" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start">
          <DropdownMenuItem className="cursor-pointer">
            Detailed
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <DropdownMenu modal={false}>
        <DropdownMenuTrigger asChild>
          <Button size="sm" variant="outline">
            Generate
            <IconChevronUpDown className="ml-1" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start">
          <DropdownMenuItem className="cursor-pointer">
            A unit test
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}
