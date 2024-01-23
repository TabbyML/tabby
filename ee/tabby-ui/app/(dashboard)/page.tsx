'use client'

import { useEffect, useState } from 'react'
import { noop } from 'lodash-es'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { useHealth } from '@/lib/hooks/use-health'
import { useMutation } from '@/lib/tabby/gql'
import { Button } from '@/components/ui/button'
import {
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle
} from '@/components/ui/card'
import { IconRotate } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { CopyButton } from '@/components/copy-button'
import SlackDialog from '@/components/slack-dialog'

export default function Home() {
  return (
    <div>
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
  const [{ data }, reexecuteQuery] = useQuery({ query: meQuery })
  const [origin, setOrigin] = useState('')
  useEffect(() => {
    setOrigin(new URL(window.location.href).origin)
  }, [])

  const resetUserAuthToken = useMutation(resetUserAuthTokenDocument, {
    onCompleted: () => reexecuteQuery()
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
          <Input value={origin} onChange={noop} className="max-w-[320px]" />
          <CopyButton value={origin} />
        </span>

        <Label>Token</Label>
        <span className="flex items-center gap-1">
          <Input
            className="max-w-[320px] font-mono text-red-600"
            value={data.me.authToken}
            onChange={noop}
          />
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
