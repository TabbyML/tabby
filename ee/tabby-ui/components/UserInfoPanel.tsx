import { useSignOut } from '@/lib/tabby/auth';
import { IconLogout } from './ui/icons';
import React from 'react';

export function UserInfoPanel({ className }: React.ComponentProps<'span'>) {
  const session = useAuthenticatedSession();
  const signOut = useSignOut();

  return session && <span className={className}>
    <span title="Sign out">
      <IconLogout className="cursor-pointer" onClick={signOut} />
    </span>
    {session.email}
  </span>;

}
