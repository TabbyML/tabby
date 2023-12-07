import { useSession, useSignOut } from '@/lib/tabby/auth'
import { cn } from '@/lib/utils'
import { IconLogout } from './ui/icons'

export default function UserPanel() {
  const { data: session, status } = useSession()
  const signOut = useSignOut()

  if (status !== 'authenticated') return
  return (
    <div className="py-4 flex justify-center text-sm font-medium">
      <span className={cn('flex items-center gap-2')}>
        <span title="Sign out">
          <IconLogout className="cursor-pointer" onClick={signOut} />
        </span>
        {session.email}
      </span>
    </div>
  )
}
