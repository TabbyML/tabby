import { HTMLAttributes } from 'react'

import { Badge, badgeVariants } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { IconEdit, IconTrash, IconUser } from '@/components/ui/icons'

interface MembershipViewProps extends HTMLAttributes<HTMLDivElement> {
  userGroupId: string
}

const data = [
  {
    id: 'a',
    email: 'a@example.com',
    isGroupAdmin: true
  },
  {
    id: 'b',
    email: 'b@example.com',
    isGroupAdmin: false
  }
]

export function MembershipView({ userGroupId }: MembershipViewProps) {
  return (
    <div className="border-b">
      {data.map(item => {
        return (
          <div
            key={item.id}
            className="pl-10 pr-3 flex items-center gap-2 py-2 hover:bg-muted/50 border-b last:border-0"
          >
            <IconUser className="shrink-0" />
            <div className="flex-1">
              <div className="flex items-center gap-2 text-sm">
                Name
                {item.isGroupAdmin ? (
                  <Badge>Group Admin</Badge>
                ) : (
                  <Badge variant="secondary">Group Member</Badge>
                )}
              </div>
              <div className="text-muted-foreground text-sm">
                test@example.com
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button className="shrink-0" variant="ghost" size="icon">
                <IconEdit />
              </Button>
              <Button
                className="shrink-0"
                variant="hover-destructive"
                size="icon"
              >
                <IconTrash />
              </Button>
            </div>
          </div>
        )
      })}
    </div>
  )
}
