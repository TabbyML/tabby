import NiceAvatar, { genConfig } from 'react-nice-avatar'

import { useMe } from '@/lib/hooks/use-me'

export const Avatar = () => {
  const [{ data }] = useMe()

  if (!data?.me?.email) return null

  const config = genConfig(data?.me?.email)

  return (
    <div className="flex h-16 w-16 rounded-full border">
      <NiceAvatar className="w-full" {...config} />
    </div>
  )
}
