import {
  paths as _paths,
  webhooks as _webhooks,
  components as _components,
  $defs as _$defs,
  external as _external,
  operations as _operations,
} from "../lib";

export interface paths extends _paths {
  "/v1/health": _paths["/v1/health"] & {
    // backward compatible for Tabby server 0.2.x and earlier
    post: operations["health"];
  };
  // backward compatible for Tabby server 0.10.x and earlier
  "/v1beta/chat/completions": {
    post: operations["chat_completions"];
  };
}

export type webhooks = _webhooks;
export type components = _components;
export type $defs = _$defs;
export type external = _external;

export interface operations extends _operations {
  event: _operations["event"] & {
    // Add a query parameter to specify the select kind
    parameters: {
      query: {
        select_kind?: string | null;
      };
    };
  };
}
