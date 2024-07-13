import {
  paths as _paths,
  webhooks as _webhooks,
  components as _components,
  $defs as _$defs,
  operations as _operations,
} from "../lib";

export interface paths extends Omit<_paths, "/v1/health"> {
  // backward compatible for Tabby server 0.2.x and earlier
  "/v1/health": Omit<_paths["/v1/health"], "post"> & {
    post: operations["health"];
  };
  // backward compatible for Tabby server 0.10.x and earlier
  "/v1beta/chat/completions": _paths["/v1/chat/completions"];
}

export type webhooks = _webhooks;
export type components = _components;
export type $defs = _$defs;

export interface operations extends Omit<_operations, "event"> {
  event: Omit<_operations["event"], "parameters"> & {
    // Add a query parameter to specify the select kind
    parameters: Omit<_operations["event"]["parameters"], "query"> & {
      query: {
        select_kind?: string | undefined | null;
      };
    };
  };
}
