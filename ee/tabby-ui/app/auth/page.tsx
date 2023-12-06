import { Metadata } from "next"

import Signup from "./components/signup"

export const metadata: Metadata = {
  title: "Authentication",
  description: "Authentication forms built using the components.",
}

export default function AuthenticationPage() {
  return (
    <div className="flex flex-col items-center justify-center flex-1">
      <Signup />
    </div>
  )
}
