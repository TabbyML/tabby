import { noop } from 'lodash-es'

import { useMe } from '@/lib/hooks/use-me'
import { Input } from '@/components/ui/input'

export const Email = () => {
  const [{ data }] = useMe()

  return (
    <div>
      <Input
        disabled
        className="w-full md:w-[350px]"
        value={data?.me?.email}
        onChange={noop}
      />
    </div>
  )
}
