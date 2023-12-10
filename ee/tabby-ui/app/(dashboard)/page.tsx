'use client'

import { useEffect, useState } from 'react'

import { graphql } from '@/lib/gql/generates'
import { useHealth } from '@/lib/hooks/use-health'
import { useAuthenticatedGraphQLQuery, useGraphQLForm } from '@/lib/tabby/gql'
import {
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle
} from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { CopyButton } from '@/components/copy-button'
import SlackDialog from '@/components/slack-dialog'
import { Button } from '@/components/ui/button'
import { IconRotate } from '@/components/ui/icons'
import { Label } from '@/components/ui/label'

export default function Home() {
  return (
    <div className="p-4 lg:p-16">
      <MainPanel />
      <SlackDialog />
    </div>
  )
}

const meQuery = graphql(/* GraphQL */ `
  query MeQuery {
    me {
      authToken
    }
  }
`)

const resetUserAuthTokenDocument = graphql(/* GraphQL */ `
  mutation ResetUserAuthToken {
    resetUserAuthToken
  }
`)

function MainPanel() {
  const { data: healthInfo } = useHealth()
  const { data, mutate } = useAuthenticatedGraphQLQuery(meQuery)
  const [origin, setOrigin] = useState('')
  useEffect(() => {
    setOrigin(new URL(window.location.href).origin)
  }, [])

  const { onSubmit: resetUserAuthToken } = useGraphQLForm(resetUserAuthTokenDocument, {
    onSuccess: () => mutate(),
  })

  if (!healthInfo || !data) return

  return (
    <div>
      <CardHeader>
        <CardTitle>Getting Started</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <Label>Endpoint URL</Label>
        <span className="flex items-center gap-1">
          <Input value={origin} className='max-w-[320px]' />
          <CopyButton value={origin} />
        </span>

        <Label>Token</Label>
        <span className="flex items-center gap-1">
          <Input className="text-red-600 font-mono max-w-[320px]" value={data.me.authToken} />
          <Button
            title="Rotate"
            size="icon"
            variant="hover-destructive"
            onClick={() => resetUserAuthToken()}
          >
            <IconRotate />
          </Button>
          <CopyButton value={data.me.authToken} />
        </span>
      </CardContent>
      <CardFooter>
        <span>
          Use informations above for IDE extensions / plugins configuration, see{' '}
          <a
            className="underline"
            target="_blank"
            href="https://tabby.tabbyml.com/docs/extensions/configurations#server"
          >
            documentation website
          </a>{' '}
          for details
        </span>
      </CardFooter>
    </div>
  )
}
