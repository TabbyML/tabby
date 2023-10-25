"use client";

import { buttonVariants } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { IconSlack } from "@/components/ui/icons"
import { Separator } from "@/components/ui/separator";
import { PropsWithChildren, useEffect, useState } from "react";

const COMMUNITY_DIALOG_SHOWN_KEY = "community-dialog-shown";

export default function IndexPage() {
  const [healthInfo, setHealthInfo] = useState<HealthInfo | undefined>();
  useEffect(() => {
    fetchHealth().then(setHealthInfo);
  }, []);

  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (!localStorage.getItem(COMMUNITY_DIALOG_SHOWN_KEY)) {
      setOpen(true);
      localStorage.setItem(COMMUNITY_DIALOG_SHOWN_KEY, "true");
    }
  }, []);

  const gettingStartedMarkDown = `
  You can find our documentation [here](https://tabby.tabbyml.com/docs/getting-started).
  - 💻 [IDE/Editor Extensions](https://tabby.tabbyml.com/docs/extensions/)
  - ⚙️ [Configuration](https://tabby.tabbyml.com/docs/configuration)`;
  return <div className="grow flex justify-center items-center">
    <div className="w-2/3 lg:w-1/3 flex flex-col gap-3">
      {healthInfo && <>
        <h1><span className="font-bold">Congratulations</span>, your tabby instance is running!</h1>
        <span className="flex gap-1">
          <a target="_blank" href={`https://github.com/TabbyML/tabby/releases/tag/${healthInfo.version.git_describe}`}><img src={`https://img.shields.io/badge/version-${toBadgeString(healthInfo.version.git_describe)}-green`} /></a>
          <img src={`https://img.shields.io/badge/device-${healthInfo.device}-blue`} />
          <img src={`https://img.shields.io/badge/model-${toBadgeString(healthInfo.model)}-red`} />
        </span>
        <Separator />

        <p>
          You can find our documentation <Link href="https://tabby.tabbyml.com/docs/getting-started">here</Link>.
          <ul className="mt-2">
            <li>💻 <Link href="https://tabby.tabbyml.com/docs/extensions/">IDE/Editor Extensions</Link></li>
            <li>⚙️ <Link href="https://tabby.tabbyml.com/docs/configuration">Configuration</Link></li>
          </ul>
        </p>
      </>
      }
    </div>
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

interface HealthInfo {
  device: string,
  model: string,
  version: {
    build_date: string,
    git_describe: string,
  }
}

async function fetchHealth(): Promise<HealthInfo> {
  if (process.env.NODE_ENV === "production") {
    const resp = await fetch("/v1/health");
    return await resp.json() as HealthInfo;
  } else {
    return {
      "device": "metal",
      "model": "TabbyML/StarCoder-1B",
      "version": {
        "build_date": "2023-10-21",
        "git_describe": "v0.3.1",
        "git_sha": "d5fdcf3a2cbe0f6b45d6e8ef3255e6a18f840132"
      }
    } as HealthInfo
  }
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