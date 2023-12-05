import { Metadata } from "next"

import { UserAuthForm } from "./components/user-auth-form"
import { useGraphQL } from "@/lib/hooks/use-graphql"
import { getIsAdminInitialized } from "@/lib/gql/request-documents"

export const metadata: Metadata = {
  title: "Authentication",
  description: "Authentication forms built using the components.",
}

export default function AuthenticationPage() {
  return (
    <div className="flex flex-col items-center justify-center flex-1">
      <div className="space-y-6 w-[350px]">
        <div className="flex flex-col space-y-2 text-center">
          <h1 className="text-2xl font-semibold tracking-tight">
            Create an account
          </h1>
          <p className="text-sm text-muted-foreground">
            Enter your credentials below to create admin account
          </p>
        </div>
        <UserAuthForm />
      </div>
    </div>
  )
}
