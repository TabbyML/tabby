#!/usr/bin/env node

/**
 * Usage: node scripts/codegen.js [--fetch [<url>]]
 * - no-fetch: run codegen only
 * - fetch: fetch openapi schema json from url, then run codegen
 */

const path = require("path");
const fs = require("fs");
const openapiTS = require("openapi-typescript");
const ts = require("typescript");

const shouldDownload = process.argv[2] === "--fetch";
const schemaUrl = new URL(process.argv[3] || "http://127.0.0.1:8080/api-docs/openapi.json");
const jsonFile = path.join(__dirname, "../openapi.json");
const dtsFile = path.join(__dirname, "../lib/tabby.d.ts");


function patchOpenAiTypes(schema) {
  schema["paths"]["/v1/chat/completions"]["post"]["requestBody"]["content"]["application/json"]["schema"] = { "$ref": "#/components/schemas/ChatCompletionRequest" };
  schema["paths"]["/v1/chat/completions"]["post"]["responses"]["200"]["content"] = { "text/event-stream": { "schema": { "$ref": "#/components/schemas/ChatCompletionChunk" } } };
  schema["components"]["schemas"]["ChatCompletionRequest"] = { "type": "object" };
  schema["components"]["schemas"]["ChatCompletionChunk"] = { "type": "object" };
  schema["components"]["schemas"]["ChatCompletionRequestMessage"] = { "type": "object" };
  return schema;
}

async function download(url, outputFile) {
  const response = await fetch(url);
  const data = await response.text();
  fs.writeFileSync(outputFile, data);
}

async function main() {
  if (shouldDownload) {
    await download(schemaUrl, jsonFile);
  }
  const schema = JSON.parse(fs.readFileSync(jsonFile, "utf-8"));
  const patched = patchOpenAiTypes(schema);
  const ast = await openapiTS.default(patched, {
    transform: (schemaObject, metadata) => {
      if (metadata.path === "#/components/schemas/ChatCompletionRequest") {
        return ts.factory.createTypeReferenceNode(ts.factory.createIdentifier("ChatCompletionRequest"));
      }
      if (metadata.path === "#/components/schemas/ChatCompletionChunk") {
        return ts.factory.createTypeReferenceNode(ts.factory.createIdentifier("ChatCompletionChunk"));
      }
      if (metadata.path === "#/components/schemas/ChatCompletionRequestMessage") {
        return ts.factory.createTypeReferenceNode(ts.factory.createIdentifier("ChatCompletionRequestMessage"));
      }
    }
  });
  const typeContents = openapiTS.astToString(ast);
  const importContents = "import { ChatCompletionRequest, ChatCompletionChunk, ChatCompletionRequestMessage } from './openai';\n";
  const contents = openapiTS.COMMENT_HEADER + "\n" + importContents + "\n" + typeContents;
  fs.writeFileSync(dtsFile, contents);

  console.log("✅ Generated types successfully.")
}

main();
