import tls from "tls";
import path from "path";
import fs from "fs-extra";
import winCa from "win-ca/api";
import * as macCa from "mac-ca";
import type { AgentConfig } from "./AgentConfig";
import { isBrowser } from "./env";
import "./ArrayExt";
import { rootLogger } from "./logger";

type Cert = string | winCa.Certificate;

const logger = rootLogger.child({ component: "CaCert" });
let extraCaCerts: Cert[] = [];
let originalCreateSecureContext: typeof tls.createSecureContext | undefined = undefined;

function appendCaCerts(certs: Cert[]) {
  if (!originalCreateSecureContext) {
    originalCreateSecureContext = tls.createSecureContext;
  }
  const filtered = certs.filter((cert) => {
    if (typeof cert === "string") {
      return cert.trim().length > 0;
    }
    return true;
  });
  const merged = [...extraCaCerts, ...filtered].distinct();
  logger.debug(`Loaded ${merged.length - extraCaCerts.length} extra certs.`);
  extraCaCerts = merged;
  tls.createSecureContext = (options) => {
    const secureContext = originalCreateSecureContext!(options);
    extraCaCerts.forEach((cert) => {
      secureContext.context.addCACert(cert);
    });
    return secureContext;
  };
}

async function loadFromFiles(files: string) {
  logger.debug(`Loading extra certs from ${files}.`);
  const certs = (
    await files.split(path.delimiter).mapAsync(async (cert) => {
      try {
        return (await fs.readFile(cert)).toString();
      } catch (err) {
        return null;
      }
    })
  )
    .join("\n")
    .split(/(?=-----BEGIN\sCERTIFICATE-----)/g)
    .distinct();
  appendCaCerts(certs);
}

export async function loadTlsCaCerts(options: AgentConfig["tls"]) {
  if (isBrowser) {
    return;
  }
  if (options.caCerts === "bundled") {
    return;
  } else if (options.caCerts === "system") {
    if (process.platform === "win32") {
      logger.debug(`Loading extra certs from win-ca.`);
      winCa.exe(path.join("win-ca", "roots.exe"));
      winCa({
        fallback: true,
        inject: "+",
      });
    } else if (process.platform === "darwin") {
      logger.debug(`Loading extra certs from mac-ca.`);
      const certs = macCa.get();
      appendCaCerts(certs);
    } else {
      // linux: load from openssl cert
      await loadFromFiles(path.join("/etc/ssl/certs/ca-certificates.crt"));
    }
  } else if (options.caCerts) {
    await loadFromFiles(options.caCerts);
  }
}
