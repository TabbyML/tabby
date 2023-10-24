import { Button, buttonVariants } from "@/components/ui/button"
import { IconSlack } from "@/components/ui/icons"

export default function IndexPage() {
  return <div className="grow flex justify-center items-center">
    <div className="w-2/3 lg:w-1/3 flex flex-col gap-3">
      <h1 className="text-xl font-bold">Join the Tabby community</h1>
      <p>Connect with other contributors building Tabby. Share knowledge, get help, and contribute to the open-source project.</p>
      <p>
        <a target="_blank" href="https://join.slack.com/t/tabbycommunity/shared_invite/zt-1xeiddizp-bciR2RtFTaJ37RBxr8VxpA" className={buttonVariants()}><IconSlack className="-ml-2 h-8 w-8" />Join us on Slack</a>
      </p>
    </div>
  </div>
}