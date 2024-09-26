import type { ConfigData } from "./config/type";
import type { Configurations } from "./config";
import tls from "tls";
import path from "path";
import fs from "fs-extra";
import winCa from "win-ca/api";
import * as macCa from "mac-ca";
import deepEqual from "deep-equal";
import "./utils/array";
import { isBrowser } from "./env";
import { getLogger } from "./logger";

type Cert = string | winCa.Certificate;

const logger = getLogger("CertsLoader");
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

async function loadTlsCaCerts(config: ConfigData["tls"]) {
  if (config.caCerts === "bundled") {
    return;
  } else if (config.caCerts === "system") {
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
  } else if (config.caCerts) {
    await loadFromFiles(config.caCerts);
  }
}

export class CertsLoader {
  constructor(private readonly configurations: Configurations) {}

  async preInitialize() {
    if (isBrowser) {
      return;
    }
    const config = this.configurations.getMergedConfig()["tls"];
    await loadTlsCaCerts(config);
    this.configurations.on("updated", async (config: ConfigData, oldConfig: ConfigData) => {
      if (!deepEqual(config["tls"], oldConfig["tls"])) {
        await loadTlsCaCerts(config["tls"]);
      }
    });
  }
}
