import { Metadata } from 'next'

import { SubHeader } from '@/components/sub-header'

export const metadata: Metadata = {
  title: 'User Group'
}

export default function UserGroupLayout({
  children
}: {
  children: React.ReactNode
}) {
  return (
    <>
      <SubHeader className="mb-8">
        Assign and regulate member access to different source contexts with User
        Groups, ensuring secure and customized interactions.
      </SubHeader>
      {children}
    </>
  )
}
