import { Metadata } from 'next'

import UserGroup from './components/user-group-page'

export const metadata: Metadata = {
  title: 'User Group'
}

export default function UserGroupPage() {
  return <UserGroup />
}
