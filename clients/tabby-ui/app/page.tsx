"use client"

import { buttonVariants } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { IconSlack } from "@/components/ui/icons"
import { Separator } from "@/components/ui/separator";
import { useHealth } from "@/lib/hooks/use-health";
import { PropsWithChildren, useEffect, useState } from "react";

const COMMUNITY_DIALOG_SHOWN_KEY = "community-dialog-shown";

export default function IndexPage() {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (!localStorage.getItem(COMMUNITY_DIALOG_SHOWN_KEY)) {
      setOpen(true);
      localStorage.setItem(COMMUNITY_DIALOG_SHOWN_KEY, "true");
    }
  }, []);

  return <div className="grow flex justify-center items-center" >
    <MainPanel />
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent>
        <DialogHeader className="gap-3">
          <DialogTitle>Join the Tabby community</DialogTitle>
          <DialogDescription>
            Connect with other contributors building Tabby. Share knowledge, get help, and contribute to the open-source project.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter className="sm:justify-start">
          <a target="_blank" href="https://join.slack.com/t/tabbycommunity/shared_invite/zt-1xeiddizp-bciR2RtFTaJ37RBxr8VxpA" className={buttonVariants()}><IconSlack className="-ml-2 h-8 w-8" />Join us on Slack</a>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  </div>
}

interface LinkProps {
  href: string
}

function Link({ href, children }: PropsWithChildren<LinkProps>) {
  return <a target="_blank" href={href} className="underline">{children}</a>
}

function toBadgeString(str: string) {
  return encodeURIComponent(str.replaceAll("-", "--"));
}

function MainPanel() {
  const { data: healthInfo } = useHealth();

  if (!healthInfo) return

  return <div className="w-2/3 lg:w-1/3 flex flex-col gap-3">
    <h1><span className="font-bold">Congratulations</span>, your tabby instance is running!</h1>
    <span className="flex flex-wrap gap-1">
      <a target="_blank" href={`https://github.com/TabbyML/tabby/releases/tag/${healthInfo.version.git_describe}`}>
        <img src={`https://img.shields.io/badge/version-${toBadgeString(healthInfo.version.git_describe)}-green`} />
      </a>
      <img src={`https://img.shields.io/badge/device-${healthInfo.device}-blue`} />
      <img src={`https://img.shields.io/badge/model-${toBadgeString(healthInfo.model)}-red`} />
      {healthInfo.chat_model && <img src={`https://img.shields.io/badge/chat%20model-${toBadgeString(healthInfo.chat_model)}-orange`} />}
    </span>

    <Separator />

    <span>
      You can find our documentation <Link href="https://tabby.tabbyml.com/docs/getting-started">here</Link>.
      <ul className="mt-2">
        <li>💻 <Link href="https://tabby.tabbyml.com/docs/extensions/">IDE/Editor Extensions</Link></li>
        <li>⚙️ <Link href="https://tabby.tabbyml.com/docs/configuration">Configuration</Link></li>
      </ul>
    </span>
  </div>
}