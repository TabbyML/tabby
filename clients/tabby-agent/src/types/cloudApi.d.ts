export interface paths {
  "/device-token": {
    post: operations["deviceToken"];
  };
  "/device-token/accept": {
    post: operations["deviceTokenAccept"];
  };
  "/device-token/refresh": {
    post: operations["deviceTokenRefresh"];
  };
  "/usage": {
    post: operations["usage"];
  };
}

export type webhooks = Record<string, never>;

export interface components {
  schemas: {
    DeviceTokenRequest: {
      auth_url: string;
    };
    DeviceTokenResponse: {
      data: {
        code: string;
      };
    };
    DeviceTokenAcceptResponse: {
      data: {
        jwt: string;
      };
    };
    DeviceTokenRefreshResponse: {
      data: {
        jwt: string;
      };
    };
    UsageRequest: object;
  };
  responses: never;
  parameters: never;
  requestBodies: never;
  headers: never;
  pathItems: never;
}

export type $defs = Record<string, never>;

export type external = Record<string, never>;

export interface operations {
  deviceToken: {
    requestBody: {
      content: {
        "application/json": components["schemas"]["DeviceTokenRequest"];
      };
    };
    responses: {
      200: {
        content: {
          "application/json": components["schemas"]["DeviceTokenResponse"];
        };
      };
    };
  };
  deviceTokenAccept: {
    parameters: {
      query: {
        code: string;
      };
    };
    responses: {
      200: {
        content: {
          "application/json": components["schemas"]["DeviceTokenAcceptResponse"];
        };
      };
    };
  };
  deviceTokenRefresh: {
    responses: {
      200: {
        content: {
          "application/json": components["schemas"]["DeviceTokenRefreshResponse"];
        };
      };
    };
  };
  usage: {
    requestBody: {
      content: {
        "application/json": components["schemas"]["UsageRequest"];
      };
    };
    responses: {
      200: {
        content: never;
      };
    };
  };
}
