'use client'

import useRouterStuff from '@/lib/hooks/use-router-stuff'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'

type DocType = 'preset' | 'custom'

interface TypeFilterProps {
  type: DocType
}

export function TypeFilter({ type }: TypeFilterProps) {
  const { updateUrlComponents, pathname } = useRouterStuff()
  const onTypeChange = (type: string) => {
    const _pathname = `/settings/providers/doc/${type}`
    if (type && pathname.startsWith(_pathname)) {
      return
    }

    updateUrlComponents({
      pathname: _pathname
    })
  }

  return (
    <Tabs value={type} onValueChange={onTypeChange}>
      <TabsList className="h-9">
        <TabsTrigger value="preset" className="h-7">
          Preset
        </TabsTrigger>
        <TabsTrigger value="custom" className="h-7">
          Custom
        </TabsTrigger>
      </TabsList>
    </Tabs>
  )
}
